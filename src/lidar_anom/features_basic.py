from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


_PCD_TYPE_MAP = {
    ("f", 4): np.float32,
    ("f", 8): np.float64,
    ("i", 1): np.int8,
    ("i", 2): np.int16,
    ("i", 4): np.int32,
    ("i", 8): np.int64,
    ("u", 1): np.uint8,
    ("u", 2): np.uint16,
    ("u", 4): np.uint32,
    ("u", 8): np.uint64,
}


def _parse_header(lines: List[bytes]) -> Tuple[Dict[str, str], int]:
    header: Dict[str, str] = {}
    header_len = 0
    for raw in lines:
        header_len += len(raw)
        line = raw.decode("ascii", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        key = parts[0].upper()
        val = parts[1] if len(parts) > 1 else ""
        header[key] = val
        if key == "DATA":
            break
    return header, header_len


def _build_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    dtype_fields = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        np_type = _PCD_TYPE_MAP.get((t.lower(), int(s)))
        if np_type is None:
            raise ValueError(f"Unsupported field type/size: field={f} type={t} size={s}")
        if int(c) == 1:
            dtype_fields.append((f, np_type))
        else:
            dtype_fields.append((f, np_type, (int(c),)))
    return np.dtype(dtype_fields)


def load_pcd_fields(pcd_path: str) -> Dict[str, Optional[np.ndarray]]:

    with open(pcd_path, "rb") as f:
        header_lines = []
        while True:
            raw = f.readline()
            if not raw:
                break
            header_lines.append(raw)
            if raw.decode("ascii", errors="ignore").strip().upper().startswith("DATA"):
                break

        header, _ = _parse_header(header_lines)
        data_mode = header.get("DATA", "").strip().lower()
        if not data_mode:
            raise ValueError(f"PCD missing DATA: {pcd_path}")

        fields = header.get("FIELDS", "").split()
        sizes = [int(x) for x in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts_str = header.get("COUNT", "").split()
        counts = [int(x) for x in counts_str] if counts_str else [1] * len(fields)

        n_points = int(header.get("POINTS", header.get("WIDTH", "0")).strip() or "0")
        if n_points <= 0:
            return {"xyz": np.zeros((0, 3), dtype=np.float32), "intensity": None}

        if data_mode == "ascii":
            try:
                ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
            except ValueError:
                raise ValueError(f"Missing x/y/z: {pcd_path}")
            ii = fields.index("intensity") if "intensity" in fields else None

            xyz = []
            intens = [] if ii is not None else None

            for raw in f:
                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    xyz.append((float(parts[ix]), float(parts[iy]), float(parts[iz])))
                    if ii is not None:
                        intens.append(float(parts[ii]))
                except Exception:
                    continue

            xyz_arr = np.asarray(xyz, dtype=np.float32) if xyz else np.zeros((0, 3), dtype=np.float32)
            inten_arr = np.asarray(intens, dtype=np.float32) if intens is not None else None
            return {"xyz": xyz_arr, "intensity": inten_arr}

        if data_mode == "binary":
            point_dtype = _build_dtype(fields, sizes, types, counts)
            point_step = point_dtype.itemsize

            data = f.read(n_points * point_step)
            if len(data) < n_points * point_step:
                n_points = len(data) // point_step
                data = data[: n_points * point_step]

            arr = np.frombuffer(data, dtype=point_dtype, count=n_points)

            if "x" not in arr.dtype.names or "y" not in arr.dtype.names or "z" not in arr.dtype.names:
                raise ValueError(f"Missing x/y/z: {pcd_path}")

            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)

            inten = None
            if "intensity" in arr.dtype.names:
                inten = np.asarray(arr["intensity"], dtype=np.float32)

            return {"xyz": xyz, "intensity": inten}

        if "binary_compressed" in data_mode:
            raise ValueError(f"binary_compressed not supported: {pcd_path}")

        raise ValueError(f"Unknown DATA mode '{data_mode}' in {pcd_path}")


def _q(x: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    if x.size == 0:
        return np.zeros(len(qs), dtype=np.float32)
    return np.quantile(x, qs).astype(np.float32)


def _radial_hist(r: np.ndarray, bins: int = 8) -> np.ndarray:
    if r.size == 0:
        return np.zeros(bins, dtype=np.float32)
    eps = 1e-6
    r99 = float(np.quantile(r, 0.99))
    r_hi = max(r99, eps)
    edges = np.linspace(0.0, r_hi, bins + 1)
    h, _ = np.histogram(np.clip(r, 0, r_hi), bins=edges)
    h = (h / max(h.sum(), 1)).astype(np.float32)
    return h


def _cov_shape_feats(xyz: np.ndarray) -> np.ndarray:
    eps = 1e-6
    mean_xyz = xyz.mean(axis=0)
    centered = xyz - mean_xyz
    cov = (centered.T @ centered) / max(xyz.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov).astype(np.float32)  # ascending
    l1, l2, l3 = float(eigvals[2]), float(eigvals[1]), float(eigvals[0])  # descending
    l1s = l1 + eps
    linearity = (l1 - l2) / l1s
    planarity = (l2 - l3) / l1s
    sphericity = l3 / l1s
    return np.array([l1, l2, l3, linearity, planarity, sphericity], dtype=np.float32)


def _iqr(a: np.ndarray) -> float:
    q25, q75 = np.percentile(a, [25, 75])
    return float(q75 - q25)


def _mad(a: np.ndarray) -> float:
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _outlier_frac_iqr(a: np.ndarray) -> float:
    q25, q75 = np.percentile(a, [25, 75])
    iqr = q75 - q25
    if iqr <= 1e-12:
        return 0.0
    lo = q25 - 1.5 * iqr
    hi = q75 + 1.5 * iqr
    return float(np.mean((a < lo) | (a > hi)))


def _polar_entropy(xyz: np.ndarray, n_az: int = 32, n_r: int = 16) -> float:

    if xyz is None or xyz.shape[0] == 0:
        return 0.0

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = np.sqrt(x*x + y*y + z*z)
    az = np.arctan2(y, x)
    az = np.where(az < 0, az + 2*np.pi, az)

    az_bins = np.floor(az / (2*np.pi) * n_az).astype(int)
    az_bins = np.clip(az_bins, 0, n_az - 1)

    r99 = np.percentile(r, 99)
    if r99 <= 1e-12:
        return 0.0
    r_clip = np.clip(r, 0, r99)
    r_bins = np.floor(r_clip / (r99 + 1e-6) * n_r).astype(int)
    r_bins = np.clip(r_bins, 0, n_r - 1)

    H = np.zeros((n_az, n_r), dtype=np.float32)
    np.add.at(H, (az_bins, r_bins), 1.0)

    p = H.reshape(-1)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def extract_features_corruption_plus(
    xyz: np.ndarray,
    intensity: Optional[np.ndarray],
    include_entropy: bool = False
) -> np.ndarray:

    if xyz is None or xyz.shape[0] == 0:
        d = 38 if include_entropy else 37
        return np.zeros(d, dtype=np.float32)

    n = float(xyz.shape[0])
    std_xyz = xyz.std(axis=0).astype(np.float32)  # 3

    r = np.linalg.norm(xyz, axis=1).astype(np.float32)
    r_mean = float(r.mean())
    r_std = float(r.std())
    r_q = _q(r, (0.1, 0.5, 0.9))                   # 3

    z = xyz[:, 2].astype(np.float32)
    z_q = _q(z, (0.1, 0.5, 0.9))                   # 3

    shape = _cov_shape_feats(xyz)                  # 6
    hist = _radial_hist(r, bins=8)                 # 8

    # +3 azimuth quantiles
    az = np.arctan2(xyz[:, 1], xyz[:, 0]).astype(np.float32)
    az_q = _q(az, (0.1, 0.5, 0.9))                  # 3

    # +1 r_q99, +1 r_max
    r_q99 = float(np.percentile(r, 99))
    r_max = float(r.max())

    # +3 robust spread for r
    r_iqr = _iqr(r)
    r_mad = _mad(r)
    r_out = _outlier_frac_iqr(r)

    # +3 robust spread for z
    z_iqr = _iqr(z)
    z_mad = _mad(z)
    z_out = _outlier_frac_iqr(z)

    extra = np.array(
        [az_q[0], az_q[1], az_q[2],
         r_q99, r_max, r_iqr, r_mad, r_out,
         z_iqr, z_mad, z_out],
        dtype=np.float32
    )

    if include_entropy:
        pent = _polar_entropy(xyz)
        extra = np.concatenate([extra, np.array([pent], dtype=np.float32)], axis=0)

    feats = np.concatenate([
        np.array([n], dtype=np.float32),              # 1
        std_xyz,                                      # 3
        np.array([r_mean, r_std], dtype=np.float32),  # 2
        r_q,                                          # 3
        z_q,                                          # 3
        shape,                                        # 6
        hist,                                         # 8
        extra,                                        # 11 or 12
    ], axis=0)

    return feats.astype(np.float32, copy=False)


def extract_features_from_pcd(pcd_path: str, feature_set: str = "corruption_plus") -> np.ndarray:
    d = load_pcd_fields(pcd_path)
    xyz = d["xyz"]
    intensity = None

    if feature_set == "corruption_plus":
        return extract_features_corruption_plus(xyz, intensity, include_entropy=False)

    raise ValueError(f"Unknown feature_set: {feature_set}")