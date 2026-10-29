import numpy as np

def _quantile_from_pmf_grid(z, p, q=0.5, interpolation=False):
    """
    Quantile for a discrete distribution with masses p on support points z.

    Parameters
    ----------
    z : array_like, shape (N,)
        Support points.
    p : array_like, shape (N,)
        Probability masses (nonnegative). The function renormalizes if sum(p) != 1.
    q : float
        Quantile level in [0, 1].
    interpolation : bool
        If False (default): return the smallest z such that CDF(z) >= q.
        If True: linearly interpolate in (z, CDF) between the two grid points that
        bracket q. This is a heuristic on the grid, not a canonical discrete quantile.
        Should be used only if underlying distribution is continuous and z is a fine grid.

    Returns
    -------
    float
        Quantile estimate.
    """
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)

    if z.ndim != 1 or p.ndim != 1 or z.size != p.size:
        raise ValueError("z and p must be 1D arrays of the same length.")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(p)):
        raise ValueError("Non-finite input.")
    if np.any(p < 0):
        raise ValueError("p must be nonnegative.")

    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Sum of masses must be positive.")
    p = p / s

    order = np.argsort(z)
    z = z[order]
    p = p[order]

    cdf = np.cumsum(p)

    if q <= 0.0:
        return float(z[0])
    if q >= 1.0:
        return float(z[-1])

    i = int(np.searchsorted(cdf, q, side="left"))
    i = min(max(i, 0), z.size - 1)

    if not interpolation or i == 0:
        return float(z[i])

    z0, z1 = z[i - 1], z[i]
    c0, c1 = cdf[i - 1], cdf[i]

    if c1 <= c0:
        return float(z1)

    t = (q - c0) / (c1 - c0)
    return float(z0 + t * (z1 - z0))

def _local_maxima(y, rtol=1e-8, atol=1e-12):
    """
    Return indices of local maxima in a 1D array.

    Rules:
      - Strict peaks: y[i] > y[i-1] and y[i] > y[i+1] with tolerance.
      - Plateaus: contiguous equal values; if higher than neighbors,
        return the integer midpoint of the plateau.
      - Edges (0 and n-1) are ignored.
    """
    y = np.asarray(y)
    n = y.size
    if n < 3:
        return np.empty(0, dtype=int)

    def gt(a, b):
        tol = max(abs(a), abs(b)) * rtol + atol
        return a > b + tol

    def ge(a, b):
        tol = max(abs(a), abs(b)) * rtol + atol
        return a >= b - tol

    peaks = []
    i = 1
    while i < n - 1:
        # strict single-point peak
        if gt(y[i], y[i - 1]) and gt(y[i], y[i + 1]):
            peaks.append(i)
            i += 1
            continue

        # potential plateau peak
        if ge(y[i], y[i - 1]) and ge(y[i], y[i + 1]) and (gt(y[i], y[i - 1]) or gt(y[i], y[i + 1])):
            j = i
            while j < n - 1 and np.isclose(y[j], y[j + 1], rtol=rtol, atol=atol):
                j += 1

            left_higher = gt(y[i], y[i - 1])
            right_lower = (j < n - 1) and gt(y[j], y[j + 1])  # guard right edge

            if left_higher and right_lower:
                peaks.append((i + j) // 2)

            i = j + 1
        else:
            i += 1

    return np.array(peaks, dtype=int)

def pmf_stats(z, p, interpolation=False):
    """
    z: array of length N (support points, z >= 0 allowed but not required)
    p: array of length N (probability masses), p >= 0, sum(p) = 1 (renormalized if needed)
    Returns mean, median, mode, std, skewness, non-excess kurtosis, and peak info.
    """
    import numpy as np

    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)

    if z.ndim != 1 or p.ndim != 1 or len(z) != len(p):
        raise ValueError("z and p must be 1D arrays of the same length.")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(p)):
        raise ValueError("Non-finite input.")
    if np.any(p < 0):
        raise ValueError("p must be non-negative.")

    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    p = p / s

    # mean and central moments (discrete)
    mean = float(np.sum(p * z))
    x = z - mean
    var = float(np.sum(p * x**2))

    eps = 1e-12
    if not np.isfinite(var) or var <= eps:
        std = 0.0
        skewness = 0.0
        kurtosis = np.nan
    else:
        std = float(np.sqrt(var))
        mu3 = float(np.sum(p * x**3))
        mu4 = float(np.sum(p * x**4))
        skewness = float(mu3 / (std**3)) if np.isfinite(mu3) else 0.0
        kurtosis = float(mu4 / (var**2)) if np.isfinite(mu4) else np.nan  # non-excess

    # median (discrete)
    median = _quantile_from_pmf_grid(z, p, q=0.5, interpolation=interpolation)

    # mode (discrete)
    mode_idx = int(np.argmax(p))
    mode = float(z[mode_idx])

    # peaks (discrete)
    peak_idx = _local_maxima(p)

    if peak_idx.size == 0:
        main_peak_height = float(p[mode_idx])
        second_peak_height = None
        peak_ratio = None
        second_peak_pos = None
    else:
        # ensure the global maximum is included among peaks
        if mode_idx not in peak_idx:
            peak_idx = np.unique(np.append(peak_idx, mode_idx))

        heights = p[peak_idx]
        order = np.argsort(heights)[::-1]
        peak_idx_sorted = peak_idx[order]
        heights_sorted = heights[order]

        main_peak_height = float(heights_sorted[0])

        if len(heights_sorted) >= 2:
            second_peak_height = float(heights_sorted[1])
            second_peak_pos = float(z[peak_idx_sorted[1]])
            peak_ratio = float(heights_sorted[1] / heights_sorted[0]) if heights_sorted[0] > 0 else None
        else:
            second_peak_height = None
            second_peak_pos = None
            peak_ratio = None

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode),
        "std": float(std),
        "skewness": float(skewness),
        "kurtosis": None if np.isnan(kurtosis) else float(kurtosis),
        "main_peak_height": float(main_peak_height),
        "second_peak_height": second_peak_height,
        "second_peak_pos": second_peak_pos,
        "peak_height_ratio": peak_ratio,
    }
