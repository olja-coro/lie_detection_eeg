import numpy as np
import pywt
from numpy.typing import NDArray
from numba import njit, prange
from math import exp, log


@njit
def fuzzy_sum_chebyshev(slices, n, r):
    K, m = slices.shape
    total = 0.0

    for i in range(K):
        for j in range(i + 1, K):
            max_diff = 0.0
            for k in range(m):
                d = abs(slices[i, k] - slices[j, k])
                if d > max_diff:
                    max_diff = d

            total += exp(-((max_diff / r) ** n))

    return 2.0 * total


@njit
def create_slices(channel_data, m):
    N = len(channel_data)
    
    if N < m:
        return np.empty((0, m), dtype=channel_data.dtype)
        
    num_vectors = N - m + 1
    
    out = np.empty((num_vectors, m), dtype=channel_data.dtype)
    
    for i in range(num_vectors):
        out[i] = channel_data[i : i + m]
        
    return out


@njit
def calculate_phi(channel_data, m, n, r):
    slices = create_slices(channel_data, m)
    K = slices.shape[0]

    if K < 2:
        return 0.0

    s = fuzzy_sum_chebyshev(slices, n, r)

    return s / (K * (K - 1))


@njit
def fuzzy_entropy(
    channel_data: NDArray[np.float64],
    r: float,
    m: int = 2,
    n: int = 2
):
    phi_m = calculate_phi(channel_data, m, n, r)    
    phi_m1 = calculate_phi(channel_data, m + 1, n, r)

    if phi_m <= 0 or phi_m1 <= 0:
        print('Returning zero fuzzy entropy')
        return 0

    return log(phi_m) - log(phi_m1)
    

@njit(parallel=True)
def tsmfe_features(
    channel_data: NDArray[np.float64], #Numba prefers explicit or inferred types, here it's okay
    r: float,
    k_max: int = 10,
    m: int = 2,
    n: int = 2,
):    
    final_results = np.empty(k_max, dtype=np.float64)

    for i in prange(k_max):
        tau = i + 1  #current scale (from 1 to k_max)

        current_tau_sum = 0.0

        for k in range(tau):
            sub_signal = channel_data[k::tau]
            
            en_val = fuzzy_entropy(sub_signal, r, m, n)
            
            if np.isnan(en_val) or np.isinf(en_val):
                val = 0.0
            else:
                val = en_val
            
            current_tau_sum += val

        final_results[i] = current_tau_sum / tau
    
    return final_results


def hmfe_features(
    channel_data: np.ndarray,
    r: float,
    m: int = 2,
    n: int = 2,
    wavelet: str = "db4",
    level: int = 4
) -> np.ndarray:
    x = np.asarray(channel_data, dtype=np.float64)
    N = x.shape[0]

    # sicurezza
    if N <= m + 1:
        return np.zeros(level + 1, dtype=np.float64)

    std_x = np.std(x).item()
    if std_x <= 0:
        std_x = 1e-8

    if r <= 0 or not np.isfinite(r):
        r = 0.15 * std_x

    # DWT
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="symmetric")
    cA = coeffs[0]          # A_L
    details = coeffs[1:]   # [D_L, D_{L-1}, ..., D_1]

    def reconstruct_band(keep_approx, detail_idx=None):
        new_coeffs = []
        # Approximation
        new_coeffs.append(cA if keep_approx else np.zeros_like(cA))
        # Details
        for i, cd in enumerate(details):
            if (not keep_approx) and (i == detail_idx):
                new_coeffs.append(cd)
            else:
                new_coeffs.append(np.zeros_like(cd))
        rec = pywt.waverec(new_coeffs, wavelet=wavelet, mode="symmetric")
        return rec[:N]

    hmfe_vals = []

    # A_L
    xA = reconstruct_band(True)
    std_A = np.std(xA).item()
    if std_A <= 0:
        std_A = 1e-8
    rA = r * (std_A / std_x)
    feA = fuzzy_entropy(xA, rA, m, n)
    hmfe_vals.append(0.0 if not np.isfinite(feA) else feA)

    # D_L ... D_1
    for idx in range(len(details)):
        xD = reconstruct_band(False, idx)
        std_D = np.std(xD)
        if std_D <= 0:
            std_D = 1e-8
        rD = r * (std_D / std_x)
        feD = fuzzy_entropy(xD, rD, m, n)
        hmfe_vals.append(0.0 if not np.isfinite(feD) else feD)

    return np.asarray(hmfe_vals, dtype=np.float64)


def extract_single_channel_features(
    channel_data,
    k_max: int = 10,
    m: int = 2,
    n: int = 2,
    r_factor: float = 0.15
):
    r = np.std(channel_data).item() * r_factor

    fe = fuzzy_entropy(channel_data, r, m, n)
    tsmfe = tsmfe_features(channel_data, r, k_max, m, n)    
    hmfe = hmfe_features(channel_data, r, m, n)
    
    return np.concatenate((tsmfe, hmfe, [fe]))