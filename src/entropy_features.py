import numpy as np
import pywt

def _embed(x: np.ndarray, m: int) -> np.ndarray:
    """
    Build m-dimensional embedding vectors from 1D signal x.
    Returns array of shape (N-m+1, m).
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N < m:
        raise ValueError(f"Signal too short for embedding m={m}, len={N}")
    return np.stack([x[i:i+m] for i in range(N - m + 1)], axis=0)


def fuzzy_entropy(x: np.ndarray, m: int = 2, r: float | None = None,
                  n: int = 2, r_factor: float = 0.15) -> float:
    """
    Fuzzy Entropy (FE) for a 1D signal x.

    Parameters
    ----------
    x : 1D array-like
        Signal.
    m : int
        Embedding dimension.
    r : float or None
        Tolerance. If None, r = r_factor * std(x).
    n : int
        Fuzzy power.
    r_factor : float
        Multiplier for std(x) if r is None.

    Returns
    -------
    fe : float
        Fuzzy entropy value.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N <= m + 1:
        # Not enough points
        return 0.0


    if r is None:
        std_x = np.std(x)
        r = r_factor * std_x if std_x > 0 else 1e-8
    if r <= 0:
        r = 1e-8


    def _phi(m_dim: int) -> float:
        Xm = _embed(x, m_dim)          # shape: (L, m_dim)
        L = Xm.shape[0]

        # Chebyshev distances between all pairs (i, j)
        count_sim = np.zeros(L)
        for i in range(L):
            # Broadcast diff to all j
            diff = np.abs(Xm[i] - Xm)
            d = np.max(diff, axis=1)   
            # Exclude self
            d[i] = np.inf

            # Fuzzy membership function μ(d)
            mu = np.exp(-(d ** n) / r)
            # Exclude self (was set to inf → exp(-inf)=0 anyway, but keep explicit)
            mu[i] = 0.0

            # Average similarity for vector i
            count_sim[i] = np.sum(mu) / (L - 1)

        # Average over all i
        return np.mean(count_sim)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    eps = 1e-12
    if phi_m1 < eps or phi_m < eps:
        return 0.0

    return -np.log(phi_m1 / phi_m)

def tsmfe(x: np.ndarray, m: int = 2, n: int = 2,
          r: float | None = None, r_factor: float = 0.15,
          k_max: int = 10) -> np.ndarray:
    """
    Time-Shifted Multi-Scale Fuzzy Entropy (TSMFE) of a 1D signal.

    For each scale k=1..k_max:
      - build k time-shifted subsequences x_j^(k)
      - compute FE on each subsequence
      - TSMFE(k) = mean FE over the k subsequences

    Returns vector of length k_max.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N <= m + 1:
        return np.zeros(k_max, dtype=float)

    if r is None:
        std_x = np.std(x)
        r = r_factor * std_x if std_x > 0 else 1e-8

    features = []

    for k in range(1, k_max + 1):
        fes = []
        for shift in range(k):
            subseq = x[shift::k]
            if subseq.size <= m + 1:
                fes.append(0.0)
            else:
                fe = fuzzy_entropy(subseq, m=m, n=n, r=r)
                fes.append(fe)
        features.append(float(np.mean(fes)))

    return np.asarray(features, dtype=float)


def hmfe(x: np.ndarray, m: int = 2, n: int = 2, r_factor: float = 0.15,
         wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """
    Hierarchical Multi-Band Fuzzy Entropy (HMFE) using DWT.

    Steps:
      - DWT: x -> [cA_L, cD_L, cD_{L-1}, ..., cD_1]
      - For each band (A_L and each D_i):
          * reconstruct band-limited signal via IDWT
          * adjust tolerance r_band = r * (std_band / std_x)
          * compute FE on band signal
      - Return [FE_A_L, FE_D_L, FE_D_{L-1}, ..., FE_D_1]
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N <= m + 1:
        # Not enough data
        return np.zeros(level + 1, dtype=float)

    std_x = np.std(x)
    if std_x == 0:
        std_x = 1e-8
    r_base = r_factor * std_x

    # DWT decomposition
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    # coeffs[0] = cA_L
    # coeffs[1] = cD_L, coeffs[2] = cD_{L-1}, ..., coeffs[level] = cD_1
    cA = coeffs[0]
    details = coeffs[1:]

    fe_bands = []

    # Helper to reconstruct with only one band non-zero
    def _reconstruct_band(keep_approx: bool, detail_index: int | None) -> np.ndarray:
        """
        keep_approx: True for A_L band, False for detail bands
        detail_index: index into 'details' (0..level-1) when keep_approx is False
        """
        new_coeffs = []
        if keep_approx:
            new_coeffs.append(cA.copy())
        else:
            new_coeffs.append(np.zeros_like(cA))

        for idx, cd in enumerate(details):
            if (not keep_approx) and (idx == detail_index):
                new_coeffs.append(cd.copy())
            else:
                new_coeffs.append(np.zeros_like(cd))

        rec = pywt.waverec(new_coeffs, wavelet=wavelet)
        # waverec can be slightly longer; truncate to original length
        return rec[:N]

    # 1) Approximation band A_L
    xA = _reconstruct_band(keep_approx=True, detail_index=None)
    std_A = np.std(xA)
    if std_A == 0:
        std_A = 1e-8
    r_A = r_base * (std_A / std_x)
    fe_A = fuzzy_entropy(xA, m=m, n=n, r=r_A)
    fe_bands.append(fe_A)

    # 2) Detail bands D_L, D_{L-1}, ..., D_1 in this order
    for idx in range(len(details)):  # idx = 0..level-1
        xD = _reconstruct_band(keep_approx=False, detail_index=idx)
        std_D = np.std(xD)
        if std_D == 0:
            std_D = 1e-8
        r_D = r_base * (std_D / std_x)
        fe_D = fuzzy_entropy(xD, m=m, n=n, r=r_D)
        fe_bands.append(fe_D)

    return np.asarray(fe_bands, dtype=float)

def extract_entropy_features(x: np.ndarray,
                             m: int = 2,
                             n: int = 2,
                             r_factor: float = 0.15,
                             k_max: int = 10,
                             wavelet: str = "db4",
                             level: int = 4) -> np.ndarray:
    """
    Extract FE + TSMFE + HMFE features for one 1D trial.

    Returns feature vector of length:
        1 (FE) + k_max (TSMFE) + (level + 1) (HMFE) = level + k_max + 2
    """
    x = np.asarray(x, dtype=float)

    # base r from full signal
    std_x = np.std(x)
    if std_x == 0:
        std_x = 1e-8
    r_base = r_factor * std_x

    fe = fuzzy_entropy(x, m=m, n=n, r=r_base)
    tsmfe_vec = tsmfe(x, m=m, n=n, r=r_base, k_max=k_max)
    hmfe_vec = hmfe(x, m=m, n=n, r_factor=r_factor,
                    wavelet=wavelet, level=level)

    return np.concatenate(([fe], tsmfe_vec, hmfe_vec), axis=0)
