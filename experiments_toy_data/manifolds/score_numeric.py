import torch
from torch.func import jacrev, vmap

## https://docs.pytorch.org/functorch/stable/generated/functorch.jacrev.html
def score_jacobian(score_fn, x, t):
    def s_one(xi, ti):
        return score_fn(xi.unsqueeze(0), ti.unsqueeze(0)).squeeze(0)
    return vmap(jacrev(s_one, argnums=0))(x, t)

## neural network jacobian not necessarily symmetric, symmetrize for toy metrics 
def score_jacobian_symmetric(score_fn, x, t):
    J = score_jacobian(score_fn, x, t)
    return 0.5 * (J + J.transpose(-1, -2))

class DiffusionDeriv:
    def __init__(self, score_fn, t):
        self.score_fn = score_fn
        self.t = float(t)

    def _t_vec(self, x):
        return x.new_full((x.shape[0],), self.t)

    def score(self, x):
        return self.score_fn(x, self._t_vec(x))

    def hessian(self, x):
        with torch.enable_grad():
            return score_jacobian_symmetric(self.score_fn, x, self._t_vec(x))


# validation 
def numerical_jacobian(score_fn, x, t, eps=1e-3):
    N, D = x.shape
    J = torch.zeros(N, D, D, device=x.device, dtype=x.dtype)
    for d in range(D):
        e = torch.zeros_like(x); e[:, d] = eps
        s_p = score_fn(x + e, t)
        s_m = score_fn(x - e, t)
        J[:, :, d] = (s_p - s_m) / (2 * eps)
    return 0.5 * (J + J.transpose(-1, -2))