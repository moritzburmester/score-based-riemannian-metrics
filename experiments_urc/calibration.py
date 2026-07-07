import torch

# Calibration strategy as described in the appendix of Bethune et al. 

def calibrate_ab(h_fn, on_manifold, off_manifold, gmin=1.0, gmax=1000.0, inverted=False):
    with torch.no_grad():
        h_on = float(h_fn(on_manifold).mean())
        h_off = float(h_fn(off_manifold).mean())
    denom = h_off - h_on
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    if inverted:
        alpha = (1.0 / gmax - 1.0 / gmin) / denom
        beta = 1.0 / gmin - alpha * h_on
    else:
        alpha = (gmax - gmin) / denom
        beta = gmin - alpha * h_on
    return alpha, beta, {"h_on": h_on, "h_off": h_off, "alpha": alpha, "beta": beta}


def make_anchors(real_t, rng, n_pairs=2000):
    i = torch.as_tensor(rng.integers(0, real_t.shape[0], n_pairs), device=real_t.device)
    j = torch.as_tensor(rng.integers(0, real_t.shape[0], n_pairs), device=real_t.device)
    on = torch.cat([real_t[i], real_t[j]], 0)
    off = 0.5 * (real_t[i] + real_t[j])
    return on, off