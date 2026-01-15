import numpy as np
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

            total += exp(-(max_diff ** n) / r)

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
def fuzzy_entropy(
    channel_data: NDArray[np.float64],
    r: float,
    m: int = 2,
    n: int = 2
):
    def phi(pattern_length):
        slices = create_slices(channel_data, pattern_length)
        K = slices.shape[0]

        if K < 2:
            return 0.0

        s = fuzzy_sum_chebyshev(slices, n, r)
        return s / (K * (K - 1))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)

    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan

    return log(phi_m) - log(phi_m1)
    

@njit(parallel=True)
def tsmfe_features(
    channel_data: NDArray[np.float64], # Numba preferisce tipi espliciti o inferiti, qui è ok
    r: float,
    k_max: int = 10,
    m: int = 2,
    n: int = 2,
):    
    final_results = np.empty(k_max, dtype=np.float64)

    for i in prange(k_max):
        tau = i + 1  # La scala attuale (da 1 a k_max)

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
    channel_data: NDArray[np.float64],
    r: float,
    m: int = 2,
    n: int = 2,
):
    # x = np.asarray(x, dtype=float)
    # N = x.shape[0]
    # if N <= m + 1:
    #     # Not enough data
    #     return np.zeros(level + 1, dtype=float)

    # std_x = np.std(x)
    # if std_x == 0:
    #     std_x = 1e-8
    # r_base = r_factor * std_x

    # # DWT decomposition
    # coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    # # coeffs[0] = cA_L
    # # coeffs[1] = cD_L, coeffs[2] = cD_{L-1}, ..., coeffs[level] = cD_1
    # cA = coeffs[0]
    # details = coeffs[1:]

    # fe_bands = []

    # # Helper to reconstruct with only one band non-zero
    # def _reconstruct_band(keep_approx: bool, detail_index: int | None) -> np.ndarray:
    #     """
    #     keep_approx: True for A_L band, False for detail bands
    #     detail_index: index into 'details' (0..level-1) when keep_approx is False
    #     """
    #     new_coeffs = []
    #     if keep_approx:
    #         new_coeffs.append(cA.copy())
    #     else:
    #         new_coeffs.append(np.zeros_like(cA))

    #     for idx, cd in enumerate(details):
    #         if (not keep_approx) and (idx == detail_index):
    #             new_coeffs.append(cd.copy())
    #         else:
    #             new_coeffs.append(np.zeros_like(cd))

    #     rec = pywt.waverec(new_coeffs, wavelet=wavelet)
    #     # waverec can be slightly longer; truncate to original length
    #     return rec[:N]

    # # 1) Approximation band A_L
    # xA = _reconstruct_band(keep_approx=True, detail_index=None)
    # std_A = np.std(xA)
    # if std_A == 0:
    #     std_A = 1e-8
    # r_A = r_base * (std_A / std_x)
    # fe_A = fuzzy_entropy(xA, m=m, n=n, r=r_A)
    # fe_bands.append(fe_A)

    # # 2) Detail bands D_L, D_{L-1}, ..., D_1 in this order
    # for idx in range(len(details)):  # idx = 0..level-1
    #     xD = _reconstruct_band(keep_approx=False, detail_index=idx)
    #     std_D = np.std(xD)
    #     if std_D == 0:
    #         std_D = 1e-8
    #     r_D = r_base * (std_D / std_x)
    #     fe_D = fuzzy_entropy(xD, m=m, n=n, r=r_D)
    #     fe_bands.append(fe_D)

    # return np.asarray(fe_bands, dtype=float)
    return [0] * 20


def extract_sigle_channel_features(
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
    
    return np.concatenate(([fe], tsmfe, hmfe))