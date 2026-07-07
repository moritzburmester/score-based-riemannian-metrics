import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def graph_init_curve(kinetic_fn, data_np, z0, z1, n_points,
                     k=15, subsample=2000, seed=0, weight_floor=1e-8):
    device = z0.device
    if subsample and data_np.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        data_np = data_np[rng.choice(data_np.shape[0], subsample, replace=False)]
    z0n = z0.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    z1n = z1.detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    V = np.concatenate([data_np.astype(np.float32), z0n, z1n], 0)
    N, D = V.shape
    qa, qb = N - 2, N - 1
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(V)
    _, idx = nn.kneighbors(V)
    src = np.repeat(np.arange(N), k)
    dst = idx[:, 1:].reshape(-1)
    p_i = torch.tensor(V[src], dtype=torch.float32, device=device)
    diff = torch.tensor(V[dst] - V[src], dtype=torch.float32, device=device)
    mid = p_i + 0.5 * diff
    with torch.no_grad():
        w = kinetic_fn(mid, diff).cpu().numpy()
    w = np.maximum(w, weight_floor)
    W = csr_matrix((w, (src, dst)), shape=(N, N))
    _, preds = shortest_path(W, directed=False, indices=qa, return_predecessors=True)
    path = [qb]
    cur = qb
    while cur != qa:
        cur = int(preds[cur])
        if cur < 0:
            raise RuntimeError(f"no path; raise k (currently {k})")
        path.append(cur)
    path.reverse()
    pab = V[path]
    seg = np.linalg.norm(np.diff(pab, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= max(s[-1], 1e-12)
    out = np.stack([np.interp(np.linspace(0, 1, n_points), s, pab[:, d])
                    for d in range(D)], -1)
    return torch.tensor(out, dtype=torch.float32)


def graph_init_batch(kinetic_fn, data_np, z0, z1, n_points, k=15, seed=0):
    return torch.stack([
        graph_init_curve(kinetic_fn, data_np, z0[i], z1[i], n_points, k=k, seed=seed)
        for i in range(z0.shape[0])
    ], 0)