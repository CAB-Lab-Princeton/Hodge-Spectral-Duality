"""
Data-adaptive basis selection.

The default basis takes the k eigenvectors of smallest eigenvalue. When the
solution measure of a task is not concentrated in the low-frequency band
(for instance when the underlying physics has a resonant response), the
same k directions can be chosen far more efficiently from data: inside each
Hodge sector, rotate to the principal directions of the training targets'
covariance, keep the harmonic directions verbatim, and merge the sectors by
carried energy. On the torus benchmark this reduces the k = 64 capture gap
from 0.49 to 0.055-0.071 relative L2 (see the project report).

Both constructions are one-time preprocessing steps per (mesh, dataset)
pair; on the benchmark meshes they take a few seconds on top of the
eigendecomposition they rotate.
"""
import numpy as np


def adaptive_basis_1form(flux_train, Phi1, labels, k, protect_harmonic=True):
    """Sector-blocked, data-adaptive basis for edge (1-form) signals.

    Args:
        flux_train: (n, E) training targets as edge fluxes
        Phi1:       (E, M) eigenbasis of the 1-form Hodge Laplacian
                    (M <= E modes; typically from HighOrderSpectralOperators
                    or operators.sector.build_sector_basis_1form)
        labels:     (M,) per-mode sector labels 'H' / 'G' / 'C'
        k:          number of basis vectors to return
        protect_harmonic: keep every harmonic direction verbatim and rank it
                    by its training energy (it is never rotated away)

    Returns:
        (Psi, labels_out, energy): (E, k) orthonormal basis whose columns
        stay inside their Hodge sector, per-column sector labels, and the
        mean training energy carried by each column (descending).
    """
    flux_train = np.asarray(flux_train, dtype=np.float64)
    Phi1 = np.asarray(Phi1, dtype=np.float64)
    n = len(flux_train)
    entries = []
    for s in ('H', 'G', 'C'):
        mask = labels == s
        if not mask.any():
            continue
        C = flux_train @ Phi1[:, mask]                    # (n, |S|)
        if s == 'H' and protect_harmonic:
            block = Phi1[:, mask]
            for jcol in range(block.shape[1]):
                entries.append((float(np.mean(C[:, jcol] ** 2)), 'H',
                                block[:, jcol]))
            continue
        _, sv, Vt = np.linalg.svd(C, full_matrices=False)
        B = Phi1[:, mask] @ Vt.T                          # stays in sector S
        en = (sv ** 2) / n
        for jcol in range(B.shape[1]):
            entries.append((float(en[jcol]), s, B[:, jcol]))
    entries.sort(key=lambda t: -t[0])
    entries = entries[:k]
    Psi = np.stack([e[2] for e in entries], axis=1)
    return (Psi, np.array([e[1] for e in entries]),
            np.array([e[0] for e in entries]))


def adaptive_basis_0form(fields_train, k):
    """Data-adaptive basis for node (0-form) signals, shared across channels.

    Args:
        fields_train: (n, N) scalar targets or (n, N, C) multi-channel
                      targets (e.g. vector fields decoded per component)
        k:            number of basis vectors

    Returns:
        (Psi, energy): (N, k) orthonormal basis and the mean training energy
        per column (descending).
    """
    Y = np.asarray(fields_train, dtype=np.float64)
    if Y.ndim == 3:
        Y = np.concatenate([Y[:, :, c] for c in range(Y.shape[2])], axis=0)
    _, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    r = min(k, len(sv))
    return Vt[:r].T, (sv[:r] ** 2) / len(Y)
