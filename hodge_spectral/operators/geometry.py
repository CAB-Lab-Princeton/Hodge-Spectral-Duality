"""
Geometry-consistent differential operators on triangle meshes.

The combinatorial Laplacians used elsewhere in this package depend only on
mesh connectivity. On meshes whose triangle shapes vary, the cotangent
(stiffness) Laplacian is the standard geometry-consistent discretization of
the Laplace-Beltrami operator: its eigenvalues carry an intrinsic
"frequency" meaning, so ordering basis functions by eigenvalue matches the
spectral content of operators defined on the surface. On the torus
benchmark this alignment roughly halves the number of modes needed for a
fixed capture level (see the project report).
"""
import numpy as np
import scipy.sparse as sp


def cotan_laplacian(points, faces, eps=1e-12):
    """Cotangent stiffness Laplacian (positive semi-definite, identity mass).

    Args:
        points: (N, 3) vertex positions
        faces:  (F, 3) triangle vertex indices
        eps:    clamp guard for degenerate triangles

    Returns:
        (N, N) sparse CSR matrix L with L @ 1 = 0, L = L^T, L >= 0.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(points)

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    # cotangent at each corner = dot / (2 * area) of the two adjacent edges
    def corner_cot(a, b, c):
        u = points[b] - points[a]
        v = points[c] - points[a]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        return np.einsum('ij,ij->i', u, v) / np.maximum(cross, eps)

    cot_i = corner_cot(i, j, k)   # opposite edge (j, k)
    cot_j = corner_cot(j, k, i)   # opposite edge (k, i)
    cot_k = corner_cot(k, i, j)   # opposite edge (i, j)

    rows = np.concatenate([j, k, k, i, i, j])
    cols = np.concatenate([k, j, i, k, j, i])
    w = 0.5 * np.concatenate([cot_i, cot_i, cot_j, cot_j, cot_k, cot_k])

    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(d) - W).tocsr()
