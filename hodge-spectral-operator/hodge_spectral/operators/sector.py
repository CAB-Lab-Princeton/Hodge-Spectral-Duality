"""
Hodge-sector-structured eigenbasis for the 1-form Laplacian.

For the UNNORMALIZED combinatorial Hodge Laplacians built from boundary
operators d0 = B1 (E, N) and d1 = B2 (F, E),

    L0 = B1^T B1,   L1 = B1 B1^T + B2^T B2,   L2 = B2 B2^T,

the spectrum of L1 decomposes exactly (d1 d0 = 0):

  * gradient sector:  every eigenpair (lam, phi) of L0 with lam > 0 gives an
    L1 eigenpair  (lam,  B1 phi / sqrt(lam))
  * curl sector:      every eigenpair (mu, psi) of L2 with mu > 0 gives an
    L1 eigenpair  (mu,   B2^T psi / sqrt(mu))
  * harmonic sector:  ker L1, dimension beta_1 (first Betti number)

so the first k eigenmodes of L1 can be assembled from the two SMALLER
eigenproblems (N x N and F x F) plus a beta_1-dimensional nullspace
computation -- the E x E eigenproblem (the largest of the three) is never
solved. As a byproduct every basis vector carries an exact sector label,
which is what makes the topological (harmonic) degrees of freedom explicitly
separable from the geometric (gradient/curl) ones.

NOTE: this identity holds for the unnormalized Laplacians only. Symmetric
degree normalization D^{-1/2} L D^{-1/2} destroys the correspondence.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, cg, minres, LinearOperator


def _small_eigs(L, k, tol_dense=2000):
    """Smallest-k eigenpairs of a sparse symmetric PSD matrix, ascending."""
    n = L.shape[0]
    k = min(k, n - 1) if n > 1 else 1
    if k <= 0:
        return np.zeros(0), np.zeros((n, 0))
    L = L.astype(np.float64)
    if n < tol_dense:
        vals, vecs = np.linalg.eigh(L.toarray())
        return vals[:k], vecs[:, :k]
    vals, vecs = eigsh(L, k=k, sigma=-1e-4, which='LM', tol=0, maxiter=5000)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def _project_out(V, B_out, L_small, rtol=1e-8, maxiter=20000):
    """Remove the component of each column of V lying in im(B_out).

    im(B_out) projection: P v = B_out pinv(L_small) (B_out^T v), where
    L_small = B_out^T B_out. The rhs B_out^T v lies in range(L_small), so a
    Krylov solve on the (singular, consistent) system yields a valid
    solution. Jacobi-preconditioned CG with a MINRES fallback (more robust
    for semidefinite systems).
    """
    W = np.asarray(V, dtype=np.float64).copy()
    d = np.asarray(L_small.diagonal()).ravel()
    d[d <= 0] = 1.0
    M = LinearOperator(L_small.shape, matvec=lambda x: x / d)
    for j in range(W.shape[1]):
        rhs = B_out.T @ W[:, j]
        x, info = cg(L_small, rhs, rtol=rtol, atol=0.0, maxiter=maxiter, M=M)
        if info != 0:
            x, info = minres(L_small, rhs, rtol=rtol, maxiter=maxiter, M=M)
            if info != 0:
                raise RuntimeError(f"CG/MINRES failed to converge (info={info})")
        W[:, j] -= B_out @ x
    return W


def harmonic_basis_1form(B1, B2, h_dim, oversample=4, seed=0, rtol=1e-10):
    """Orthonormal basis of ker L1 (harmonic 1-cochains) without factorizing L1.

    im(B1), im(B2^T) and ker L1 are mutually orthogonal, so a single
    projection sweep of a random matrix followed by QR recovers the kernel.
    One polish sweep controls CG round-off.
    """
    E = B1.shape[0]
    if h_dim <= 0:
        return np.zeros((E, 0))
    L0 = (B1.T @ B1).tocsr()
    L2 = (B2 @ B2.T).tocsr()
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((E, h_dim + oversample))
    for _ in range(2):  # projection + one polish pass
        V = _project_out(V, B1, L0, rtol=rtol)
        V = _project_out(V, B2.T, L2, rtol=rtol)
        V, _ = np.linalg.qr(V)
    # keep the h_dim directions best aligned with ker L1 (smallest residual)
    L1V = B1 @ (B1.T @ V) + B2.T @ (B2 @ V)
    res = np.linalg.norm(L1V, axis=0)
    keep = np.argsort(res)[:h_dim]
    H, _ = np.linalg.qr(V[:, np.sort(keep)])
    return H


def build_sector_basis_1form(B1, B2, k, null_tol=1e-8, seed=0, logger=None):
    """First-k eigenbasis of the unnormalized L1, assembled sector-by-sector.

    Args:
        B1: (E, N) sparse signed incidence (d0)
        B2: (F, E) sparse signed incidence (d1)
        k:  number of L1 eigenmodes wanted
        null_tol: eigenvalues below this count as zero modes

    Returns dict with:
        Phi1   (E, k) orthonormal eigenvectors, ascending eigenvalue
        lam1   (k,)   eigenvalues
        labels (k,)   'H' harmonic / 'G' gradient / 'C' curl
        beta   (beta0, beta1, beta2_like) rank-based Betti numbers
    """
    B1 = B1.tocsr().astype(np.float64)
    B2 = B2.tocsr().astype(np.float64)
    E, N = B1.shape
    F = B2.shape[0]
    log = logger.log if logger else print

    L0 = (B1.T @ B1).tocsr()
    L2 = (B2 @ B2.T).tocsr()

    pool = k + 4
    lam0, Phi0 = _small_eigs(L0, pool)
    lam2, Phi2 = _small_eigs(L2, pool)

    z0 = int(np.sum(lam0 < null_tol))          # beta0 (connected components)
    z2 = int(np.sum(lam2 < null_tol))          # dim ker(d1 d1^T) = dim ker(d1^T)
    beta1 = E - (N - z0) - (F - z2)            # E - rank(B1) - rank(B2)
    log(f"[Sector] beta0={z0}, beta1={beta1}, ker(L2)={z2} "
        f"(N={N}, E={E}, F={F})")

    g_idx = np.where(lam0 >= null_tol)[0]
    c_idx = np.where(lam2 >= null_tol)[0]
    U_g = (B1 @ Phi0[:, g_idx]) / np.sqrt(lam0[g_idx])[None, :]
    U_c = (B2.T @ Phi2[:, c_idx]) / np.sqrt(lam2[c_idx])[None, :]

    try:
        H = harmonic_basis_1form(B1, B2, beta1, seed=seed)
    except RuntimeError as e:
        log(f"[Sector][Warn] projection-based harmonic failed ({e}); "
            f"falling back to shift-invert eigsh on L1")
        L1 = (B1 @ B1.T + B2.T @ B2).tocsr()
        _, H = eigsh(L1, k=max(beta1, 1), sigma=-1e-6, which='LM', tol=0)
        H = H[:, :beta1]

    entries = [(0.0, 'H', H[:, i]) for i in range(H.shape[1])]
    entries += [(lam0[i], 'G', U_g[:, j]) for j, i in enumerate(g_idx)]
    entries += [(lam2[i], 'C', U_c[:, j]) for j, i in enumerate(c_idx)]
    entries.sort(key=lambda t: (t[0], t[1]))   # ascending lam; G before C on ties

    if len(entries) < k:
        log(f"[Sector][Warn] only {len(entries)} modes available (asked {k})")
    entries = entries[:k]

    Phi1 = np.stack([e[2] for e in entries], axis=1) if entries else np.zeros((E, 0))
    lam1 = np.array([e[0] for e in entries])
    labels = np.array([e[1] for e in entries])
    return {'Phi1': Phi1, 'lam1': lam1, 'labels': labels,
            'beta': (z0, beta1, z2)}


def validate_sector_basis(B1, B2, Phi1, lam1):
    """Numerical health of an assembled basis: eigen-residual + orthonormality."""
    L1v = B1 @ (B1.T @ Phi1) + B2.T @ (B2 @ Phi1)
    resid = np.linalg.norm(L1v - Phi1 * lam1[None, :], axis=0)
    gram = Phi1.T @ Phi1
    ortho_err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    return {'max_eig_residual': float(np.max(resid)) if len(resid) else 0.0,
            'ortho_error': float(ortho_err)}
