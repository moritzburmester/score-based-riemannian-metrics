import os
import sys
import pickle
import argparse
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "..", "diffusion_model_dependencies"))

from discrete_geodesics import discrete_geodesic, init_path
from graph_initialization import graph_init_batch
from calibration import calibrate_ab, make_anchors
import h_utils
import riemannian_metrics as rm
from ebm import EBM_MLP

ckpt_ebm = "model_checkpoints/ebm_final.model"
ckpt_diff = "model_checkpoints/score_model_logs/checkpoints/best/epoch=5759--eval_loss_epoch=0.021.ckpt"
cfg_diff = "model_checkpoints/score_model_logs/config.pkl"
latents_path = "dataset/latents.npy"
stats_path = "dataset/latents_stats.npz"
n_angles = 360
device = "cuda" if torch.cuda.is_available() else "cpu"
lam_tag = {0.0: "lam00", 0.1: "lam01", 0.25: "lam025", 0.5: "lam05", 0.75: "lam075", 1.0: "lam10"}
lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
pt_root = "pt_files_urc"


def load_real():
    # load latents
    z = np.load(latents_path).astype(np.float32)
    z = z.reshape(z.shape[0], -1) if z.ndim == 2 else z.reshape(-1, z.shape[-1])
    if float(np.abs(z).max()) > 10.0:
        s = np.load(stats_path)
        z = (z - float(s["mean"])) / float(s["std"])
    return z


def load_ebm():
    # load ebm model
    net = EBM_MLP().to(device)
    ck = torch.load(ckpt_ebm, map_location=device, weights_only=False)
    net.load_state_dict(ck["weight"])
    net.eval()
    return net


def load_score_fn():
    # load score-based model
    from lightning_modules.utils import create_lightning_module
    from models.utils import get_score_fn
    cfg = pickle.load(open(cfg_diff, "rb"))
    pl = create_lightning_module(cfg)
    pl = type(pl).load_from_checkpoint(ckpt_diff)
    pl.config = cfg
    pl.configure_sde(cfg)
    pl = pl.to(device).eval()
    ck = torch.load(ckpt_diff, map_location="cpu")
    ema = (ck.get("optimizer_states") or [{}])[0].get("ema", None)
    params = list(pl.score_model.parameters())
    if ema is not None and len(ema) == len(params):
        with torch.no_grad():
            for p, e in zip(params, ema):
                p.data.copy_(e.to(p.device))
    raw = get_score_fn(pl.sde, pl.score_model, conditional=False, train=False, continuous=True)

    def score_fn(x):
        return raw(x, torch.full((x.shape[0],), 0.1, device=x.device))
    return score_fn


def sample_endpoints(real, n, gap_min, gap_max, rng):
    # sample enpoints between 10/180 degrees
    letters = rng.integers(0, 7, n)
    a0 = rng.integers(0, n_angles, n)
    steps = rng.integers(gap_min, gap_max + 1, n)
    a1 = (a0 + steps) % n_angles
    z0 = torch.tensor(real[letters * n_angles + a0], dtype=torch.float32, device=device)
    z1 = torch.tensor(real[letters * n_angles + a1], dtype=torch.float32, device=device)
    return z0, z1, letters, a0, steps


def perturb(z0, z1, sigma, real, seed):
    # perturb endpoints off-manifold if noise > 0 
    if sigma == 0:
        return z0, z1
    lat_std = float(real.std())
    gen = torch.Generator(device=z0.device).manual_seed(seed + 123)
    z0 = z0 + sigma * lat_std * torch.randn(z0.shape, generator=gen, device=z0.device)
    z1 = z1 + sigma * lat_std * torch.randn(z1.shape, generator=gen, device=z1.device)
    return z0, z1


def energy_metric(net, real_t, rng):
    # ebm metric from bethune et al 
    on, off = make_anchors(real_t, rng)
    a, b, _ = calibrate_ab(lambda x: net(x).squeeze(-1), on, off, gmin=1.0, gmax=1000.0)

    def kin(x, v):
        g = torch.clamp(a * net(x).squeeze(-1) + b, min=1.0)
        return g * v.pow(2).sum(-1)
    return kin


def rbf_metric(real_t, k=300, kappa=0.75):
    # rbf metric
    head = h_utils.h_diag_RBF(n_centers=k, latent_size=64, ambiant_size=64,
                              data_to_fit_ambiant=real_t, data_to_fit_latent=real_t,
                              kappa=kappa).to(device)
    head.normalize(real_t)
    metric = rm.ConformalMetric(head)

    def kin(x, v):
        return metric.kinetic(x[:, None, :], v[:, None, :])
    return kin


def land_metric(real_t, sigma=0.4):
    # land metrics 
    head = h_utils.h_diag_Land(reference_sample=real_t, gamma=sigma).to(device)
    head.normalize(real_t)
    metric = rm.DiagonalMetric(head)

    def kin(x, v):
        return metric.kinetic(x[:, None, :], v[:, None, :])
    return kin


def score_geodesic(score_fn, z0, z1, n_points, n_iter, lam, init, z_init, conf_const):
    # geodesic optimization for the score-based metrics from thesis 
    # does not fit in the interface provided by bethune et al, so own optimizaion function
    n_paths, dim = z0.shape
    n = n_points
    coef = 0.5 / (1.0 / (n - 1))
    if z_init is None:
        z_init = init_path(z0, z1, n, init)
    z_i = z_init[:, 1:-1].clone().detach().requires_grad_(True)
    with torch.no_grad():
        s0 = score_fn(z0)[:, None, :]
        s1 = score_fn(z1)[:, None, :]

    def path_scores(zi):
        si = score_fn(zi.reshape(-1, dim)).reshape(n_paths, n - 2, dim)
        return torch.cat([s0, si, s1], dim=1)

    if lam != 0.0:
        # calculate energy on initial path
        with torch.no_grad():
            sa = path_scores(z_i)
            ds0 = sa[:, 1:] - sa[:, :-1]
            v0 = z_init[:, 1:] - z_init[:, :-1]
            w0 = torch.ones(n_paths, n - 1, device=z0.device) if conf_const else sa[:, :-1].pow(2).sum(-1)
            t_dir = ds0.pow(2).sum(-1).sum(1).clamp_min(1e-12)
            t_norm = (w0 * v0.pow(2).sum(-1)).sum(1).clamp_min(1e-12)
    else:
        # divide by one, if we have pure jacobian or pure magnitude metric
        t_dir = torch.ones(n_paths, device=z0.device)
        t_norm = torch.ones(n_paths, device=z0.device)

    opt = torch.optim.Adam([z_i], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=1e-4)
    for _ in range(n_iter):
        opt.zero_grad()
        s = path_scores(z_i)
        x = torch.cat([z0[:, None], z_i, z1[:, None]], dim=1)
        # jac score approx. term 
        ds = s[:, 1:] - s[:, :-1]
        e_dir = ds.pow(2).sum(-1)
        if lam != 0.0:
            v = x[:, 1:] - x[:, :-1]
            # optional constant term, else score magnitude squared
            w = torch.ones_like(e_dir) if conf_const else s[:, :-1].pow(2).sum(-1)
            e_norm = w * v.pow(2).sum(-1)
            per = (1 - lam) * e_dir.sum(1) / t_dir + lam * e_norm.sum(1) / t_norm
        else:
            per = e_dir.sum(1) / t_dir
        loss = coef * per.sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_i], max_norm=1e3)
        opt.step()
        sched.step()
    with torch.no_grad():
        return torch.cat([z0[:, None], z_i, z1[:, None]], dim=1).detach().cpu()


def save(folder, cfg, paths, letters, a0, steps):
    d = os.path.join(pt_root, folder)
    os.makedirs(d, exist_ok=True)
    torch.save({"paths": paths, "letters": letters, "a0": a0, "steps": steps, "config": cfg},
               os.path.join(d, f"geo_{cfg}.pt"))


def main(args):
    rng = np.random.default_rng(args.seed)
    real = load_real()
    real_t = torch.tensor(real, dtype=torch.float32, device=device)
    z0, z1, letters, a0, steps = sample_endpoints(real, args.n_geodesics,
                                                  args.gap_min, args.gap_max, rng)

    def save_raw(folder, za, zb):
        for m in ["lerp", "slerp"]:
            p = init_path(za, zb, args.n_points, m).detach().cpu()
            save(folder, f"raw_{m}", p, letters, a0, steps)

    def run(kin, folder, name, za, zb):
        for init in args.inits:
            zi = graph_init_batch(kin, real, za, zb, args.n_points).to(device) if init == "graph" else None
            paths = discrete_geodesic(kin, za, zb, n_points=args.n_points,
                                      n_iter=args.iters, init=init, z_init=zi)
            save(folder, f"{name}_{init}", paths, letters, a0, steps)

    rng_cal = np.random.default_rng(args.seed + 7)

    if "ebm" in args.which:
        net = load_ebm()
        for noise, folder in [(0.0, "ebm"), (0.25, "ebm_noise")]:
            za, zb = perturb(z0, z1, noise, real, args.seed)
            save_raw(folder, za, zb)
            run(energy_metric(net, real_t, rng_cal), folder, "Etheta", za, zb)

    if "rbf_land" in args.which:
        for noise, folder in [(0.0, "rbf_land"), (0.25, "rbf_land_noise")]:
            za, zb = perturb(z0, z1, noise, real, args.seed)
            save_raw(folder, za, zb)
            run(rbf_metric(real_t), folder, "RBF", za, zb)
            run(land_metric(real_t), folder, "LAND", za, zb)

    if "score" in args.which:
        score_fn = load_score_fn()

        def make_score_kin(score_fn, lam, conf_const=False):
            # helper function for graph initialization
            def kin(x, v):
                s = score_fn(x)
                s_plus = score_fn(x + 0.5*v)
                s_minus = score_fn(x - 0.5*v)
                e_dir = (s_plus - s_minus).pow(2).sum(-1)
                w = torch.ones(x.shape[0], device=x.device) if conf_const else s.pow(2).sum(-1)
                e_norm = w * v.pow(2).sum(-1)
                return (1-lam)*e_dir + lam*e_norm
            return kin

        for init in args.inits:
            folder = "score_graph" if init == "graph" else "score_lerp_slerp"
            save_raw(folder, z0, z1)
            for lam in lambdas:
                zi = graph_init_batch(score_kin, real, z0, z1, args.n_points).to(device) if init == "graph" else None
                paths = score_geodesic(score_fn, z0, z1, args.n_points, args.iters, lam, init, zi, False)
                tag = f"lerp_{lam_tag[lam]}" if init == "graph" else f"{init}_{lam_tag[lam]}"
                save(folder, tag, paths, letters, a0, steps)

        save_raw("score_graph_const", z0, z1)
        for lam in lambdas:
            zi = graph_init_batch(score_kin, real, z0, z1, args.n_points).to(device)
            paths = score_geodesic(score_fn, z0, z1, args.n_points, args.iters, lam, "graph", zi, True)
            save("score_graph_const", f"lerp_{lam_tag[lam]}", paths, letters, a0, steps)

        za, zb = perturb(z0, z1, 0.25, real, args.seed)
        save_raw("score_graph_noise", za, zb)
        save_raw("score_graph_const_noise", za, zb)
        for lam in lambdas:
            zi = graph_init_batch(score_kin, real, za, zb, args.n_points).to(device)
            paths = score_geodesic(score_fn, za, zb, args.n_points, args.iters, lam, "graph", zi, False)
            save("score_graph_noise", f"graph_{lam_tag[lam]}", paths, letters, a0, steps)
            zi = graph_init_batch(score_kin, real, za, zb, args.n_points).to(device)
            paths = score_geodesic(score_fn, za, zb, args.n_points, args.iters, lam, "graph", zi, True)
            save("score_graph_const_noise", f"graph_{lam_tag[lam]}_cc", paths, letters, a0, steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["ebm", "rbf_land", "score"])
    p.add_argument("--inits", nargs="+", default=["lerp", "slerp", "graph"])
    p.add_argument("--n_geodesics", type=int, default=100)
    p.add_argument("--n_points", type=int, default=100)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--gap_min", type=int, default=10)
    p.add_argument("--gap_max", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)