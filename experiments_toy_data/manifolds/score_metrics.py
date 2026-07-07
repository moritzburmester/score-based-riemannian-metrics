import torch 
from .base import BaseManifold, _eye, normalize_metric
from .score_analytic import AnalyticDeriv 

def _G(points, deriv, lam=0.1):
    # proposed metric, in form JTJ
    H = deriv.hessian(points)
    HtH = H @ H.transpose(-1, -2) # HHT or HtH (since its symmetric)
    sn2 = deriv.score(points).pow(2).sum(-1)
    N, D = points.shape
    I = _eye(N, D, points.device, points.dtype)
    #T_H = T_s = 1.0
    A = HtH / T_H 
    B = (sn2[:, None, None] * I) / T_s
    return (1 - lam) * A + lam * B 

def _G_quad(points, v, deriv, lam=0.1):
    # proposed metric, with analytical Jacobian-vector product Jv
    H = deriv.hessian(points)                      # N, D, D
    Hv = torch.einsum('nij,nj->ni', H, v)          # N, D
    sn2 = deriv.score(points).pow(2).sum(-1)       # N,
    #T_H = T_s = 1.0 
    A = Hv.pow(2).sum(-1) / T_H  # Jacobian vector product, more efficient                
    B = sn2 * v.pow(2).sum(-1) / T_s                
    return (1 - lam) * A + lam * B

## metrics from other literature 

def _INVP(points, deriv, logp_floor=-15.0):
    # inverse-probability metric: G = (1/p)  * I 
    N, D = points.shape
    I = _eye(N, D, points.device, points.dtype)
    logp = deriv.logp(points).clamp(min=logp_floor)
    scale = torch.exp(-logp)
    return scale[:, None, None] * I
    

def _SAI(points, deriv, tau=1e-3):
    # G = HTH
    H = deriv.hessian(points)
    HtH = H @ H.transpose(-1, -2)
    N, D = points.shape
    return HtH + tau * _eye(N, D, points.device, points.dtype)


def _AZE(points, deriv, lam=1.0):
    # Azeglio (2025) 
    # G = I + lam * ssT
    # Penalizes normal direction movement
    s = deriv.score(points)
    sst = s.unsqueeze(-1) * s.unsqueeze(-2)
    N, D = points.shape
    return _eye(N, D, points.device, points.dtype) + lam * sst


def _PER(points, deriv):
    # Perone (2024)
    # G = I − ssT / (1 +||s||^2)
    # Penalizes tangent movement
    s = deriv.score(points)
    sst = s.unsqueeze(-1) * s.unsqueeze(-2)
    s_norm2 = s.pow(2).sum(-1)                                
    denom = (1.0 + s_norm2).unsqueeze(-1).unsqueeze(-1)
    N, D = points.shape
    return _eye(N, D, points.device, points.dtype) - sst / denom
