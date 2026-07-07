import torch


def lerp(z0, z1, u):
    return (1 - u) * z0 + u * z1


def slerp(z0, z1, u):
    n0 = z0 / z0.norm(dim=1, keepdim=True)
    n1 = z1 / z1.norm(dim=1, keepdim=True)
    dot = (n0 * n1).sum(1).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    a = (torch.sin((1 - u) * omega) / so)[:, None]
    b = (torch.sin(u * omega) / so)[:, None]
    return torch.where(so[:, None] > 1e-6, a * z0 + b * z1, lerp(z0, z1, u))


def init_path(z0, z1, N, mode):
    f = slerp if mode == "slerp" else lerp
    us = torch.linspace(0, 1, N, device=z0.device)
    return torch.stack([f(z0, z1, float(u)) for u in us], dim=1)


def discrete_geodesic(kinetic_fn, z0, z1, *, n_points, n_iter,
                      lr=1e-3, lr_min=1e-4, init="lerp", z_init=None):
    # batched geodesic energy minimization
    P, D = z0.shape
    N = n_points
    coef = 0.5 / (1.0 / (N - 1))
    if z_init is None:
        z_init = init_path(z0, z1, N, init)
    z_i = z_init[:, 1:-1].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z_i], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr_min)
    for ep in range(n_iter):
        opt.zero_grad()
        x = torch.cat([z0[:, None], z_i, z1[:, None]], dim=1)
        v = x[:, 1:] - x[:, :-1]
        mid = 0.5 * (x[:, 1:] + x[:, :-1])
        e = kinetic_fn(mid.reshape(-1, D), v.reshape(-1, D)).reshape(P, N - 1)
        loss = coef * e.sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_i], max_norm=1e3)
        opt.step()
        sched.step()
    with torch.no_grad():
        return torch.cat([z0[:, None], z_i, z1[:, None]], dim=1).detach().cpu()