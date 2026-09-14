"""Basic correctness tests on a minimal closed mesh (tetrahedron boundary).

Run with:  python -m pytest tests/  (or  python tests/test_basic.py)
"""
import numpy as np
import torch

from hodge_spectral.operators import (HighOrderSpectralOperators,
                                      cotan_laplacian,
                                      build_sector_basis_1form,
                                      validate_sector_basis,
                                      adaptive_basis_0form)
from hodge_spectral.models.unified import UnifiedHSD

POINTS = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                   [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
FACES = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])


def _host(**kw):
    return HighOrderSpectralOperators(POINTS, FACES, k_list=(3, 5, 3),
                                      normalize_laplacian=False, **kw)


def test_complex_counts_and_spectrum_union():
    h = _host()
    assert (h.n_nodes, h.n_edges, h.n_faces) == (4, 6, 4)
    L0 = (h.B1.T @ h.B1).toarray()
    L1 = (h.B1 @ h.B1.T + h.B2.T @ h.B2).toarray()
    L2 = (h.B2 @ h.B2.T).toarray()
    pos = lambda L: np.sort([v for v in np.linalg.eigvalsh(L) if v > 1e-8])
    union = np.sort(np.concatenate([pos(L0), pos(L2)]))
    assert np.allclose(pos(L1), union, atol=1e-8), \
        "positive spectrum of L1 must equal the union of L0 and L2 spectra"


def test_eigenvalues_sorted_and_cached():
    h = _host()
    for lam in (h.lam0, h.lam1, h.lam2):
        assert np.all(np.diff(lam) >= -1e-10)
    assert len(h.lam1) == h.Phi1.shape[1]


def test_sector_basis():
    h = _host()
    sec = build_sector_basis_1form(h.B1, h.B2, k=4)
    beta0, beta1, _ = sec['beta']
    assert (beta0, beta1) == (1, 0)          # closed genus-0 surface
    v = validate_sector_basis(h.B1, h.B2, sec['Phi1'], sec['lam1'])
    assert v['max_eig_residual'] < 1e-8
    assert v['ortho_error'] < 1e-8
    assert set(sec['labels']) <= {'H', 'G', 'C'}


def test_cotan_laplacian():
    L = cotan_laplacian(POINTS, FACES)
    A = L.toarray()
    assert np.allclose(A, A.T, atol=1e-12)
    assert np.allclose(A @ np.ones(4), 0.0, atol=1e-10)
    assert np.linalg.eigvalsh(A).min() > -1e-9


def test_l0_mode_cotan():
    h = _host(l0_mode="cotan")
    assert abs(h.lam0[0]) < 1e-8             # constant mode
    assert h.lam0[-1] > 0


def test_adaptive_basis_0form():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((12, 4))
    Psi, energy = adaptive_basis_0form(Y, 3)
    assert Psi.shape == (4, 3)
    assert np.allclose(Psi.T @ Psi, np.eye(3), atol=1e-10)
    assert np.all(np.diff(energy) <= 1e-12)


def test_model_forward():
    torch.manual_seed(0)
    h = HighOrderSpectralOperators(POINTS, FACES, k_list=(3, 5, 3))
    k0, k1, k2 = h.Phi0.shape[1], h.Phi1.shape[1], h.Phi2.shape[1]
    Phi_out = torch.from_numpy(h.Phi0.astype(np.float32))
    model = UnifiedHSD(h, 0, k0, k1, k2, h.n_nodes, k0, Phi_out,
                       hidden_dims=(16, 16), res_hidden=8)
    pred, base, res = model(torch.randn(5, k0), torch.randn(5, k1),
                            torch.randn(5, k2), torch.randn(5, h.n_nodes))
    assert pred.shape == (5, h.n_nodes)
    assert torch.isfinite(pred).all()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
