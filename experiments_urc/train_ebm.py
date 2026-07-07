import os
import argparse
import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from ebm import EBM_MLP
from sampler import init_random, get_sample_q


def load_latents(latents_path, stats_path):
    z = np.load(latents_path).astype(np.float32)
    z = z.reshape(z.shape[0], -1) if z.ndim == 2 else z.reshape(-1, z.shape[-1])
    if float(np.abs(z).max()) > 10.0:
        s = np.load(stats_path)
        z = (z - float(s["mean"])) / float(s["std"])
    return torch.from_numpy(z).float()


def main(args):
    device = args.device
    z_t = load_latents(args.latents, args.stats)
    dl = data_utils.DataLoader(
        data_utils.TensorDataset(z_t),
        batch_size=args.batch_size, shuffle=True, drop_last=True)

    netE = EBM_MLP().to(device)
    replay_buffer = init_random((args.buffer_size, 64), init_type=args.init_type)
    sample_q = get_sample_q(args)
    opt = torch.optim.Adam(netE.parameters(), lr=args.lr_init)

    path = os.path.join(args.save_root, args.model_name)
    os.makedirs(path, exist_ok=True)

    def save_ckpt(tag, ep):
        was = next(netE.parameters()).device
        netE.cpu()
        torch.save({"weight": netE.state_dict(), "type": type(netE),
                    "epoch": ep, "args": vars(args)},
                   os.path.join(path, f"{tag}.model"))
        netE.to(was)

    for ep in range(args.epoch):
        # implements CD algorithm from Bethune et al. 
        netE.train()
        pbar = tqdm(dl, leave=False)
        for (latent,) in pbar:
            latent = latent.to(device)
            z_real = latent + args.sigma * torch.randn_like(latent)
            z_q, _, replay_buffer, _ = sample_q(netE, replay_buffer, clip=args.gradient_clip)
            fp = netE(z_real)
            fq = netE(z_q)
            reg = (fp ** 2).mean() + (fq ** 2).mean()
            loss = (fp.mean() - fq.mean()) + args.w_regul * reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(netE.parameters(), max_norm=1)
            opt.step()
            pbar.set_description(f"{ep+1}/{args.epoch} loss {loss.item():.3f}")

        if ep % args.save_every == 0:
            save_ckpt(f"ep_{ep}", ep)
        save_ckpt("last", ep)
    save_ckpt("last", ep)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default="latentsnpy") # fix directory
    p.add_argument("--stats", default="latents_stats.npz")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epoch", type=int, default=20000) # stopped after 5500
    p.add_argument("--lr_init", type=float, default=1e-4)
    p.add_argument("--w_regul", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=5e-2)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--sgld_lr", type=float, default=1.0)
    p.add_argument("--sgld_std", type=float, default=1e-2)
    p.add_argument("--reinit_freq", type=float, default=0.05)
    p.add_argument("--buffer_size", type=int, default=10000)
    p.add_argument("--init_type", default="normal_01")
    p.add_argument("--gradient_clip", type=float, default=1)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--save_root", default="./ebm_runs")
    p.add_argument("--model_name", default="urc_ebm_cd")
    args = p.parse_args()
    main(args)