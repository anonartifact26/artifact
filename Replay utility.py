# %%
import os
import json
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================
# Global config for GeoLife
# =========================

GEOLIFE_ROOT = r"E:\python\Geolife Trajectories 1.3\Geolife Trajectories 1.3\Data"   # 改成你的路径
OUT_DIR = "figures_exp3_geolife"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- runtime control: keep it small enough for ~10 min ----
MAX_USERS = None                 # e.g. 20; None = all users
MAX_TRAJ_PER_USER = 3            # 每个用户最多抽几条
MIN_LEN = 80                     # 至少多少个点
MAX_LEN = 1200                   # 最多保留多少点，太长就裁掉
TARGET_NUM_TRAJ = 60             # 总共目标轨迹数；控制运行时间
SAVE_EVERY = 10                  # 每处理多少条保存一次中间统计

# ---- experiment constants: match Experiment 1 as much as possible ----
DT = 1.0
BUDGET_B = 6.0

# Static GI
STATIC_INTERVAL_S = 10.0
STATIC_EPS = 0.06

# Twinkle
TW_BASELINE_INTERVAL_S = 30.0
TW_BURST_INTERVAL_S = 2.0
TW_BURST_LEN_S = 60.0
TW_EPS_BASE = 0.03
TW_EPS_BURST = 0.12

# Cooling
TAU_COOL_S = 90.0
GEOFENCE_RADIUS_M = 30.0

# Synthetic burst placement rule
BURST_RELATIVE_POSITIONS = (0.25, 0.55, 0.80)

# Device capability simulation
SCALE_L1_ONLY   = 4.0
SCALE_L1_L5     = 2.5
SCALE_L1_L5_ADR = 1.5
NLOS_BIAS = np.array([0.8, -1.5])

RANDOM_SEED = 1234
rng_global = np.random.default_rng(RANDOM_SEED)

# %%
# ====================
# WGS84 -> ECEF -> ENU
# ====================

a = 6378137.0
f = 1 / 298.257223563
e2 = 2 * f - f * f

def wgs84_to_ecef(lat_deg, lon_deg, h_m):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    X = (N + h_m) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h_m) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + h_m) * np.sin(lat_rad)
    return X, Y, Z

def ecef_to_enu(X, Y, Z, lat0_deg, lon0_deg, h0_m):
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    X0, Y0, Z0 = wgs84_to_ecef(lat0_deg, lon0_deg, h0_m)

    dx = X - X0
    dy = Y - Y0
    dz = Z - Z0

    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    t = np.array([
        [-sin_lon,             cos_lon,            0.0],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,   cos_lat * sin_lon,  sin_lat],
    ])

    enu = t @ np.vstack((dx, dy, dz))
    return enu[0], enu[1], enu[2]

def latlon_to_local_enu(lat, lon, h):
    X, Y, Z = wgs84_to_ecef(lat, lon, h)
    lat0, lon0, h0 = lat[0], lon[0], h[0]
    E, N, U = ecef_to_enu(X, Y, Z, lat0, lon0, h0)
    return np.column_stack([E, N])

# %%
# =========================
# GeoLife trajectory loader
# =========================

def read_plt_file(path: str) -> Optional[pd.DataFrame]:
    """
    GeoLife .plt format:
    first 6 lines are header
    then: lat, lon, zero, altitude, days, date, time
    """
    try:
        df = pd.read_csv(
            path,
            skiprows=6,
            header=None,
            names=["lat", "lon", "zero", "altitude_ft", "days", "date", "time"]
        )
        if df.empty:
            return None

        # GeoLife altitude is often in feet; convert to meters
        df["altitude_m"] = pd.to_numeric(df["altitude_ft"], errors="coerce") * 0.3048
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon", "altitude_m"]).reset_index(drop=True)
        if len(df) == 0:
            return None
        return df[["lat", "lon", "altitude_m"]]
    except Exception:
        return None

def collect_geolife_files(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (user_id, plt_path)
    """
    out = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"GeoLife root not found: {root}")

    user_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if MAX_USERS is not None:
        user_dirs = user_dirs[:MAX_USERS]

    for user_id in user_dirs:
        traj_dir = os.path.join(root, user_id, "Trajectory")
        if not os.path.isdir(traj_dir):
            continue
        files = sorted([f for f in os.listdir(traj_dir) if f.lower().endswith(".plt")])

        if len(files) == 0:
            continue

        if MAX_TRAJ_PER_USER is not None and len(files) > MAX_TRAJ_PER_USER:
            files = random.sample(files, MAX_TRAJ_PER_USER)

        for fn in files:
            out.append((user_id, os.path.join(traj_dir, fn)))
    return out

def load_sampled_geolife_trajectories(root: str,
                                      target_num: int = TARGET_NUM_TRAJ,
                                      min_len: int = MIN_LEN,
                                      max_len: int = MAX_LEN) -> List[Dict]:
    """
    Load a small subset of valid trajectories for fast experiments.
    """
    all_files = collect_geolife_files(root)
    random.shuffle(all_files)

    trajs = []
    for user_id, path in all_files:
        df = read_plt_file(path)
        if df is None or len(df) < min_len:
            continue

        # crop for speed
        if len(df) > max_len:
            # keep prefix to preserve continuity
            df = df.iloc[:max_len].copy()

        x_true = latlon_to_local_enu(
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
            df["altitude_m"].to_numpy()
        )

        # basic sanity: remove degenerate tracks
        span = np.linalg.norm(x_true[-1] - x_true[0])
        if not np.isfinite(span):
            continue

        trajs.append({
            "user_id": user_id,
            "path": path,
            "n": len(df),
            "x_true": x_true,
        })

        if len(trajs) >= target_num:
            break

    return trajs

# %%
# ==========================
# Capability tier simulation
# ==========================

def simulate_internal_estimate(X_t,
                               capability: str = "L1-only",
                               add_bias: bool = True,
                               seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T, D = X_t.shape

    if capability == "L1-only":
        noise_scale = SCALE_L1_ONLY
    elif capability == "L1+L5":
        noise_scale = SCALE_L1_L5
    elif capability == "L1+L5+ADR":
        noise_scale = SCALE_L1_L5_ADR
    else:
        raise ValueError(f"Unknown capability: {capability}")

    noise = rng.laplace(loc=0.0, scale=noise_scale, size=(T, D))
    bias = NLOS_BIAS if add_bias else 0.0
    return X_t + noise + bias

# %%
# ============================
# Planar Laplace mechanism
# ============================

def planar_laplace_noise(eps: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if eps <= 0:
        raise ValueError("eps must be > 0")
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    r = rng.gamma(shape=2.0, scale=1.0 / eps, size=n)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack([dx, dy])

def planar_laplace_release(x: np.ndarray, eps_eff: float, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x + planar_laplace_noise(eps_eff, 1, rng)[0]
    else:
        return x + planar_laplace_noise(eps_eff, x.shape[0], rng)

# %%
# ============================
# Utility helper functions
# ============================

def make_intent_bursts(T: int, dt: float, burst_len_s: float = 60.0, starts: Optional[List[int]] = None):
    L = int(round(burst_len_s / dt))
    is_burst = np.zeros(T, dtype=bool)

    if starts is None:
        starts = [int(r * T) for r in BURST_RELATIVE_POSITIONS]

    for s in starts:
        s = max(0, min(T - 1, s))
        is_burst[s:s + L] = True
    return is_burst

def should_emit(t: int, last_emit_t: Optional[int], interval_steps: int) -> bool:
    if last_emit_t is None:
        return True
    return (t - last_emit_t) >= interval_steps

def circle_contains(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = np.linalg.norm(points - center[None, :], axis=1)
    return d <= radius

def compute_cooling_indicator(x_path: np.ndarray,
                              geofences: List[Tuple[np.ndarray, float]],
                              dt: float,
                              tau_cool_s: float) -> np.ndarray:
    T = x_path.shape[0]
    tau_steps = int(round(tau_cool_s / dt))
    in_any = np.zeros(T, dtype=bool)

    for (c, r) in geofences:
        in_any |= circle_contains(x_path, c, r)

    exit_times = []
    for t in range(1, T):
        if in_any[t - 1] and (not in_any[t]):
            exit_times.append(t)

    c_t = np.zeros(T, dtype=bool)
    for e in exit_times:
        c_t[e:e + tau_steps] = True
    return c_t

# %%
# ============================
# Auditor
# ============================

@dataclass
class AuditorParams:
    vmax_mps: float = 15.0
    state_stride: int = 5
    transition_slack: float = 1.25

    place_hist_s: float = 120.0
    place_r0: float = 30.0

    window_s: float = 120.0
    link_tau: float = 40.0
    background_k: int = 15

    w_route: float = 0.4
    w_place: float = 0.4
    w_link: float = 0.2
    risk_tau: float = 0.6
    gamma: float = 5.0
    alpha_max: float = 25.0

class TwinkleAuditor:
    def __init__(self,
                 x_reference: np.ndarray,
                 dt: float,
                 geofence_centers: List[np.ndarray],
                 params: AuditorParams,
                 rng: np.random.Generator):
        self.dt = dt
        self.p = params
        self.rng = rng

        self.S = x_reference[::self.p.state_stride].copy()
        self.Ns = self.S.shape[0]

        self.belief = np.ones(self.Ns) / self.Ns
        self.last_release_t = None

        self.pois = geofence_centers[:] if len(geofence_centers) > 0 else [x_reference[0]]
        self.sensitive_poi = self.pois[0]

        self.released_times = []
        self.released_points = []
        self.released_eps = []

        self.bg_phis = self._make_background_fingerprints()
        self.phi_star = None

    def _make_background_fingerprints(self):
        bg = []
        for _ in range(self.p.background_k):
            mu = self.rng.normal(0.0, 200.0, size=2)
            sig = np.abs(self.rng.normal(30.0, 10.0, size=2))
            phi = np.array([mu[0], mu[1], sig[0], sig[1]], dtype=float)
            bg.append(phi)
        return bg

    def _hmm_update_route_risk(self, z_t: np.ndarray, eps_eff: float, t: int) -> float:
        if self.last_release_t is None:
            delta_t = self.dt
        else:
            delta_t = (t - self.last_release_t) * self.dt

        thr = self.p.vmax_mps * delta_t * self.p.transition_slack
        prev = self.belief
        next_prior = np.zeros_like(prev)

        dmat = np.linalg.norm(self.S[:, None, :] - self.S[None, :, :], axis=2)
        feasible = (dmat <= thr)

        for i in range(self.Ns):
            js = np.where(feasible[i])[0]
            if js.size == 0:
                next_prior[i] += prev[i]
            else:
                next_prior[js] += prev[i] / js.size

        dist = np.linalg.norm(self.S - z_t[None, :], axis=1)
        emit = np.exp(-eps_eff * dist)

        post = next_prior * emit
        s = post.sum()
        if s <= 0:
            self.belief = np.ones(self.Ns) / self.Ns
        else:
            self.belief = post / s

        self.last_release_t = t
        return float(self.belief.max())

    def _place_risk_proxy(self, t: int) -> float:
        if len(self.released_times) == 0:
            return 0.0

        lookback_steps = int(round(self.p.place_hist_s / self.dt))
        tmin = t - lookback_steps
        idx = [k for k, tk in enumerate(self.released_times) if tk >= tmin]
        if len(idx) == 0:
            return 0.0

        Z = np.stack([self.released_points[k] for k in idx], axis=0)
        eps = np.array([self.released_eps[k] for k in idx], dtype=float)

        weights = []
        for poi in self.pois:
            d = np.linalg.norm(Z - poi[None, :], axis=1)
            w = np.exp(-np.sum(eps * d))
            weights.append(w)

        weights = np.array(weights, dtype=float)
        s = weights.sum()
        if s <= 0:
            return 0.0

        post = weights / s
        return float(np.clip(post[0], 0.0, 1.0))

    def _fingerprint_for_window(self, points: np.ndarray) -> np.ndarray:
        mu = points.mean(axis=0)
        sd = points.std(axis=0)
        return np.array([mu[0], mu[1], sd[0], sd[1]], dtype=float)

    def _link_risk_proxy(self, t: int) -> float:
        if len(self.released_times) == 0:
            return 0.0

        win_steps = int(round(self.p.window_s / self.dt))
        tmin = t - win_steps
        idx = [k for k, tk in enumerate(self.released_times) if tk >= tmin]
        if len(idx) < 3:
            return 0.0

        W = np.stack([self.released_points[k] for k in idx], axis=0)
        phi = self._fingerprint_for_window(W)

        if self.phi_star is None:
            self.phi_star = phi.copy()
            return 0.0

        def sim(phi_a, phi_b):
            l1 = np.abs(phi_a - phi_b).sum()
            return float(np.exp(-l1 / self.p.link_tau))

        s_star = sim(phi, self.phi_star)
        s_bg = max(sim(phi, b) for b in self.bg_phis)
        return float(np.clip(s_star - s_bg, 0.0, 1.0))

    def compute_risk_and_alpha(self, t: int, z_t: Optional[np.ndarray], eps_eff: float):
        if z_t is None:
            r_route = 0.0
            r_place = self._place_risk_proxy(t)
            r_link = self._link_risk_proxy(t)
        else:
            r_route = self._hmm_update_route_risk(z_t, eps_eff, t)
            self.released_times.append(t)
            self.released_points.append(z_t.copy())
            self.released_eps.append(float(eps_eff))

            r_place = self._place_risk_proxy(t)
            r_link = self._link_risk_proxy(t)

        wsum = self.p.w_route + self.p.w_place + self.p.w_link
        w1, w2, w3 = self.p.w_route / wsum, self.p.w_place / wsum, self.p.w_link / wsum
        R = float(np.clip(w1 * r_route + w2 * r_place + w3 * r_link, 0.0, 1.0))

        alpha = 1.0 + self.p.gamma * max(0.0, R - self.p.risk_tau)
        alpha = float(np.clip(alpha, 1.0, self.p.alpha_max))

        return float(r_route), float(r_place), float(r_link), R, alpha

# %%
# ============================
# Twinkle / Static GI
# ============================

@dataclass
class TwinkleParams:
    dt: float = 1.0
    baseline_interval_s: float = 30.0
    burst_interval_s: float = 2.0
    burst_len_s: float = 60.0

    eps_base: float = 0.03
    eps_burst: float = 0.12
    budget_B: float = 6.0

    tau_cool_s: float = 90.0
    cooling_mode: str = "hard"
    alpha_cool: float = 8.0
    geofence_radius_m: float = 30.0
    geofence_indices: Tuple[int, ...] = ()

    use_auditor: bool = True
    auditor_params: AuditorParams = field(default_factory=AuditorParams)

    release_center: str = "xhat"

@dataclass
class StaticGIParams:
    dt: float = 1.0
    interval_s: float = 10.0
    eps: float = 0.06
    budget_B: float = 6.0
    release_center: str = "xhat"

def simulate_static_gi(x_true: np.ndarray,
                       x_hat: np.ndarray,
                       params: StaticGIParams,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = x_true.shape[0]
    interval_steps = int(round(params.interval_s / params.dt))

    z = np.full((T, 2), np.nan)
    a = np.zeros(T, dtype=int)
    budget_used = 0.0
    last_emit_t = None
    E_budget = np.zeros(T, dtype=float)

    x_center = x_true if params.release_center == "truth" else x_hat

    for t in range(T):
        emit = should_emit(t, last_emit_t, interval_steps)
        if emit:
            remaining = params.budget_B - budget_used
            if remaining >= params.eps:
                z_t = planar_laplace_release(x_center[t], params.eps, rng)
                z[t] = z_t
                a[t] = 1
                last_emit_t = t
                budget_used += params.eps
        E_budget[t] = budget_used

    return pd.DataFrame({
        "t": np.arange(T) * params.dt,
        "xE": x_true[:, 0], "xN": x_true[:, 1],
        "xhatE": x_hat[:, 0], "xhatN": x_hat[:, 1],
        "zE": z[:, 0], "zN": z[:, 1],
        "a": a,
        "eps_eff": params.eps,
        "budget_used": E_budget,
    })

def simulate_twinkle(x_true: np.ndarray,
                     x_hat: np.ndarray,
                     params: TwinkleParams,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = x_true.shape[0]
    dt = params.dt

    # burst schedule
    is_burst = make_intent_bursts(T, dt, params.burst_len_s)

    base_steps = int(round(params.baseline_interval_s / dt))
    burst_steps = int(round(params.burst_interval_s / dt))

    # if no fixed geofence indices, place by same relative positions as burst anchors
    geo_idx = params.geofence_indices
    if len(geo_idx) == 0:
        geo_idx = tuple(int(r * T) for r in BURST_RELATIVE_POSITIONS[:2])

    geofences = []
    poi_centers = []
    for idx in geo_idx:
        idx = int(np.clip(idx, 0, T - 1))
        center = x_true[idx].copy()
        geofences.append((center, params.geofence_radius_m))
        poi_centers.append(center)

    c_t = compute_cooling_indicator(x_true, geofences, dt, params.tau_cool_s)

    auditor = None
    if params.use_auditor:
        auditor = TwinkleAuditor(
            x_reference=x_true,
            dt=dt,
            geofence_centers=poi_centers if len(poi_centers) > 0 else [x_true[0]],
            params=params.auditor_params,
            rng=rng
        )

    z = np.full((T, 2), np.nan, dtype=float)
    a = np.zeros(T, dtype=int)
    m = is_burst.astype(int)

    alpha = np.ones(T, dtype=float)
    eps_nom = np.zeros(T, dtype=float)
    eps_eff = np.zeros(T, dtype=float)
    E_budget = np.zeros(T, dtype=float)

    r_route = np.zeros(T, dtype=float)
    r_place = np.zeros(T, dtype=float)
    r_link = np.zeros(T, dtype=float)
    R_total = np.zeros(T, dtype=float)

    last_emit_t = None
    budget_used = 0.0

    x_center = x_true if params.release_center == "truth" else x_hat

    for t in range(T):
        eps_nom_t = (1 - m[t]) * params.eps_base + m[t] * params.eps_burst
        eps_nom[t] = eps_nom_t

        interval_steps = burst_steps if m[t] == 1 else base_steps
        cadence_ok = should_emit(t, last_emit_t, interval_steps)
        in_cooling = bool(c_t[t])

        alpha_t = 1.0
        eps_eff_t = eps_nom_t

        if auditor is not None:
            z_prov = planar_laplace_release(x_center[t], eps_eff_t, rng)
            rr, rp, rl, R, alpha_suggested = auditor.compute_risk_and_alpha(
                t=t, z_t=z_prov, eps_eff=eps_eff_t
            )
            r_route[t], r_place[t], r_link[t], R_total[t] = rr, rp, rl, R
            alpha_t = alpha_suggested

        if params.cooling_mode == "soft" and in_cooling:
            alpha_t = max(alpha_t, params.alpha_cool)

        alpha[t] = alpha_t
        eps_eff_t = eps_nom_t / alpha_t
        eps_eff[t] = eps_eff_t

        remaining = params.budget_B - budget_used

        def try_raise_alpha_to_fit_budget(alpha_current: float) -> float:
            if remaining <= 0:
                return alpha_current
            alpha_needed = eps_nom_t / remaining
            return float(np.clip(
                max(alpha_current, alpha_needed),
                1.0,
                params.auditor_params.alpha_max if params.use_auditor else 1e9
            ))

        emit = cadence_ok
        if params.cooling_mode == "hard" and in_cooling:
            emit = False

        if emit and remaining < eps_eff_t:
            alpha_new = try_raise_alpha_to_fit_budget(alpha_t)
            eps_eff_t2 = eps_nom_t / alpha_new
            if remaining >= eps_eff_t2:
                alpha_t = alpha_new
                eps_eff_t = eps_eff_t2
                alpha[t] = alpha_t
                eps_eff[t] = eps_eff_t
            else:
                emit = False

        if emit:
            z_t = planar_laplace_release(x_center[t], eps_eff_t, rng)
            z[t] = z_t
            a[t] = 1
            last_emit_t = t
            budget_used += eps_eff_t

            if auditor is not None:
                rr, rp, rl, R, _ = auditor.compute_risk_and_alpha(t=t, z_t=z_t, eps_eff=eps_eff_t)
                r_route[t], r_place[t], r_link[t], R_total[t] = rr, rp, rl, R

        E_budget[t] = budget_used

    return pd.DataFrame({
        "t": np.arange(T) * dt,
        "xE": x_true[:, 0], "xN": x_true[:, 1],
        "xhatE": x_hat[:, 0], "xhatN": x_hat[:, 1],
        "zE": z[:, 0], "zN": z[:, 1],
        "a": a,
        "mode": m,
        "alpha": alpha,
        "eps_nom": eps_nom,
        "eps_eff": eps_eff,
        "budget_used": E_budget,
        "cooling": c_t.astype(int),
        "r_route": r_route,
        "r_place": r_place,
        "r_link": r_link,
        "R_total": R_total,
    })

# %%
# ============================
# Metrics & plotting
# ============================

def emission_errors(df: pd.DataFrame) -> np.ndarray:
    dfe = df[df["a"] == 1].copy()
    if len(dfe) == 0:
        return np.array([])
    z = dfe[["zE", "zN"]].to_numpy()
    x = dfe[["xE", "xN"]].to_numpy()
    return np.linalg.norm(z - x, axis=1)

def emission_errors_burst_only(df: pd.DataFrame) -> np.ndarray:
    if "mode" not in df.columns:
        return np.array([])
    dfe = df[(df["a"] == 1) & (df["mode"] == 1)].copy()
    if len(dfe) == 0:
        return np.array([])
    z = dfe[["zE", "zN"]].to_numpy()
    x = dfe[["xE", "xN"]].to_numpy()
    return np.linalg.norm(z - x, axis=1)

def emission_errors_in_twinkle_bursts(df_candidate: pd.DataFrame, df_twinkle_ref: pd.DataFrame) -> np.ndarray:
    if "mode" not in df_twinkle_ref.columns:
        return np.array([])
    burst_idx = df_twinkle_ref.index[df_twinkle_ref["mode"] == 1]
    dfe = df_candidate.loc[burst_idx]
    dfe = dfe[dfe["a"] == 1].copy()
    if len(dfe) == 0:
        return np.array([])
    z = dfe[["zE", "zN"]].to_numpy()
    x = dfe[["xE", "xN"]].to_numpy()
    return np.linalg.norm(z - x, axis=1)

def summarize_errors(err: np.ndarray) -> Dict[str, float]:
    if err.size == 0:
        return {"n": 0, "median": np.nan, "p95": np.nan, "mean": np.nan}
    return {
        "n": int(err.size),
        "median": float(np.median(err)),
        "p95": float(np.quantile(err, 0.95)),
        "mean": float(np.mean(err)),
    }

def plot_cdf(errors: Dict[str, np.ndarray], title: str, out_path: str):
    plt.figure(figsize=(7, 5))
    for name, e in errors.items():
        e = np.asarray(e)
        if e.size == 0:
            continue
        e = np.sort(e)
        y = np.linspace(0, 1, e.size, endpoint=True)
        plt.plot(
            e, y,
            label=f"{name} (med={np.median(e):.2f}m, p95={np.quantile(e,0.95):.2f}m)"
        )
    plt.grid(True)
    plt.xlabel("Position error ||z - x|| (m)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# %%
# ============================
# Core experiment runners
# ============================

def run_single_traj_all_methods(x_true: np.ndarray, seed_base: int = 0):
    """
    Match Experiment 1 settings on one GeoLife trajectory.
    Returns dict of dataframes.
    """
    xhat_l1 = simulate_internal_estimate(x_true, "L1-only", seed=seed_base + 1)
    xhat_l1l5 = simulate_internal_estimate(x_true, "L1+L5", seed=seed_base + 2)
    xhat_l5adr = simulate_internal_estimate(x_true, "L1+L5+ADR", seed=seed_base + 3)

    static_params = StaticGIParams(
        dt=DT,
        interval_s=STATIC_INTERVAL_S,
        eps=STATIC_EPS,
        budget_B=BUDGET_B,
        release_center="xhat",
    )

    tw_no = TwinkleParams(
        dt=DT,
        baseline_interval_s=TW_BASELINE_INTERVAL_S,
        burst_interval_s=TW_BURST_INTERVAL_S,
        burst_len_s=TW_BURST_LEN_S,
        eps_base=TW_EPS_BASE,
        eps_burst=TW_EPS_BURST,
        budget_B=BUDGET_B,
        tau_cool_s=TAU_COOL_S,
        cooling_mode="hard",
        geofence_radius_m=GEOFENCE_RADIUS_M,
        geofence_indices=(),
        use_auditor=False,
        release_center="xhat",
    )

    tw_yes = TwinkleParams(
        dt=DT,
        baseline_interval_s=TW_BASELINE_INTERVAL_S,
        burst_interval_s=TW_BURST_INTERVAL_S,
        burst_len_s=TW_BURST_LEN_S,
        eps_base=TW_EPS_BASE,
        eps_burst=TW_EPS_BURST,
        budget_B=BUDGET_B,
        tau_cool_s=TAU_COOL_S,
        cooling_mode="hard",
        alpha_cool=8.0,
        geofence_radius_m=GEOFENCE_RADIUS_M,
        geofence_indices=(),
        use_auditor=True,
        auditor_params=AuditorParams(risk_tau=0.6, gamma=5.0, alpha_max=25.0),
        release_center="xhat",
    )

    # Main comparison uses L1-only, exactly like your Exp1 section
    df_static = simulate_static_gi(x_true, xhat_l1, static_params, seed=seed_base + 10)
    df_tw_no = simulate_twinkle(x_true, xhat_l1, tw_no, seed=seed_base + 11)
    df_tw_yes = simulate_twinkle(x_true, xhat_l1, tw_yes, seed=seed_base + 12)

    # fixed-eps across device tiers
    fixed_static = {
        "L1-only": simulate_static_gi(x_true, xhat_l1, static_params, seed=seed_base + 20),
        "L1+L5": simulate_static_gi(x_true, xhat_l1l5, static_params, seed=seed_base + 21),
        "L1+L5+ADR": simulate_static_gi(x_true, xhat_l5adr, static_params, seed=seed_base + 22),
    }

    fixed_tw_auditor = {
        "L1-only": simulate_twinkle(x_true, xhat_l1, tw_yes, seed=seed_base + 30),
        "L1+L5": simulate_twinkle(x_true, xhat_l1l5, tw_yes, seed=seed_base + 31),
        "L1+L5+ADR": simulate_twinkle(x_true, xhat_l5adr, tw_yes, seed=seed_base + 32),
    }

    # eps scaling across capability
    scales = {"L1-only": 1.0, "L1+L5": 1.5, "L1+L5+ADR": 2.0}
    scaled_tw = {}
    xhat_map = {"L1-only": xhat_l1, "L1+L5": xhat_l1l5, "L1+L5+ADR": xhat_l5adr}

    for k, scale in scales.items():
        tw_scaled = TwinkleParams(
            dt=DT,
            baseline_interval_s=TW_BASELINE_INTERVAL_S,
            burst_interval_s=TW_BURST_INTERVAL_S,
            burst_len_s=TW_BURST_LEN_S,
            eps_base=TW_EPS_BASE * scale,
            eps_burst=TW_EPS_BURST * scale,
            budget_B=BUDGET_B,
            tau_cool_s=TAU_COOL_S,
            cooling_mode="hard",
            alpha_cool=8.0,
            geofence_radius_m=GEOFENCE_RADIUS_M,
            geofence_indices=(),
            use_auditor=True,
            auditor_params=AuditorParams(risk_tau=0.6, gamma=5.0, alpha_max=25.0),
            release_center="xhat",
        )
        scaled_tw[k] = simulate_twinkle(x_true, xhat_map[k], tw_scaled, seed=seed_base + 40 + len(scaled_tw))

    return {
        "main": {
            "Static GI": df_static,
            "Twinkle (no auditor)": df_tw_no,
            "Twinkle (auditor)": df_tw_yes,
        },
        "fixed_static": fixed_static,
        "fixed_tw_auditor": fixed_tw_auditor,
        "scaled_tw": scaled_tw,
    }

# %%
# ============================
# Aggregation helpers
# ============================

def append_errors(store: Dict[str, List[np.ndarray]], key: str, arr: np.ndarray):
    if key not in store:
        store[key] = []
    if arr.size > 0:
        store[key].append(arr)

def flatten_error_store(store: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in store.items():
        if len(v) == 0:
            out[k] = np.array([])
        else:
            out[k] = np.concatenate(v, axis=0)
    return out

# %%
# ============================
# Main experiment
# ============================

start_time = time.time()
random.seed(RANDOM_SEED)

trajs = load_sampled_geolife_trajectories(
    GEOLIFE_ROOT,
    target_num=TARGET_NUM_TRAJ,
    min_len=MIN_LEN,
    max_len=MAX_LEN
)

print(f"Loaded {len(trajs)} GeoLife trajectories for Experiment 3.")

if len(trajs) == 0:
    raise RuntimeError("No valid trajectories found. Check GeoLife path / filters.")

# aggregated stores
agg_main_all = {}
agg_main_burst = {}
agg_fixed_static = {}
agg_fixed_tw = {}
agg_scaled = {}

meta_rows = []

for i, tr in enumerate(trajs):
    x_true = tr["x_true"]
    res = run_single_traj_all_methods(x_true, seed_base=1000 + i * 100)

    # main
    df_static = res["main"]["Static GI"]
    df_tw_no = res["main"]["Twinkle (no auditor)"]
    df_tw_yes = res["main"]["Twinkle (auditor)"]

    # all emissions
    append_errors(agg_main_all, "Static GI", emission_errors(df_static))
    append_errors(agg_main_all, "Twinkle (no auditor)", emission_errors(df_tw_no))
    append_errors(agg_main_all, "Twinkle (auditor)", emission_errors(df_tw_yes))

    # burst-window restricted, same as your Experiment 1b fairness version
    append_errors(agg_main_burst, "Static GI", emission_errors_in_twinkle_bursts(df_static, df_tw_no))
    append_errors(agg_main_burst, "Twinkle (no auditor)", emission_errors_in_twinkle_bursts(df_tw_no, df_tw_no))
    append_errors(agg_main_burst, "Twinkle (auditor)", emission_errors_in_twinkle_bursts(df_tw_yes, df_tw_no))

    # fixed-eps static
    for tier, df in res["fixed_static"].items():
        append_errors(agg_fixed_static, f"Static GI ({tier})", emission_errors(df))

    # fixed-eps twinkle auditor
    for tier, df in res["fixed_tw_auditor"].items():
        append_errors(agg_fixed_tw, f"Twinkle+Auditor ({tier})", emission_errors(df))

    # scaled eps
    scale_map = {"L1-only": "1.0", "L1+L5": "1.5", "L1+L5+ADR": "2.0"}
    for tier, df in res["scaled_tw"].items():
        append_errors(agg_scaled, f"Twinkle+Auditor ({tier}, eps×{scale_map[tier]})", emission_errors(df))

    # per-trajectory record (方便检查)
    meta_rows.append({
        "traj_idx": i,
        "user_id": tr["user_id"],
        "n_points": tr["n"],
        "path": tr["path"],
        "n_emit_static": int(df_static["a"].sum()),
        "n_emit_tw_no": int(df_tw_no["a"].sum()),
        "n_emit_tw_yes": int(df_tw_yes["a"].sum()),
    })

    if (i + 1) % SAVE_EVERY == 0:
        pd.DataFrame(meta_rows).to_csv(os.path.join(OUT_DIR, "exp3_geolife_progress_meta.csv"), index=False)
        print(f"Processed {i+1}/{len(trajs)} trajectories ... elapsed {time.time() - start_time:.1f}s")

# flatten
agg_main_all = flatten_error_store(agg_main_all)
agg_main_burst = flatten_error_store(agg_main_burst)
agg_fixed_static = flatten_error_store(agg_fixed_static)
agg_fixed_tw = flatten_error_store(agg_fixed_tw)
agg_scaled = flatten_error_store(agg_scaled)

# %%
# ============================
# Plot figures (same style as Exp1)
# ============================

plot_cdf(
    agg_main_all,
    title="Experiment 3 (GeoLife) — Position error CDF at all emission times",
    out_path=os.path.join(OUT_DIR, "exp3_cdf_l1_all_geolife.png")
)

plot_cdf(
    agg_main_burst,
    title="Experiment 3 (GeoLife) — Position error CDF restricted to burst windows",
    out_path=os.path.join(OUT_DIR, "exp3_cdf_l1_burst_geolife.png")
)

plot_cdf(
    agg_fixed_static,
    title="Experiment 3 (GeoLife) — Static GI, fixed epsilon across device tiers",
    out_path=os.path.join(OUT_DIR, "exp3_cdf_fixed_eps_staticGI_geolife.png")
)

plot_cdf(
    agg_fixed_tw,
    title="Experiment 3 (GeoLife) — Twinkle+Auditor, fixed epsilon across device tiers",
    out_path=os.path.join(OUT_DIR, "exp3_cdf_fixed_eps_twinkleAuditor_geolife.png")
)

plot_cdf(
    agg_scaled,
    title="Experiment 3 (GeoLife) — Twinkle+Auditor with capability-aware epsilon scaling",
    out_path=os.path.join(OUT_DIR, "exp3_cdf_eps_scaled_across_devices_geolife.png")
)

# %%
# ============================
# Save statistics for paper
# ============================

paper_rows = []

# 1) main all
for name, arr in agg_main_all.items():
    s = summarize_errors(arr)
    paper_rows.append({
        "group": "l1_all",
        "method": name,
        **s
    })

# 2) main burst
for name, arr in agg_main_burst.items():
    s = summarize_errors(arr)
    paper_rows.append({
        "group": "l1_burst",
        "method": name,
        **s
    })

# 3) fixed-eps static
for name, arr in agg_fixed_static.items():
    s = summarize_errors(arr)
    paper_rows.append({
        "group": "fixed_eps_static",
        "method": name,
        **s
    })

# 4) fixed-eps twinkle auditor
for name, arr in agg_fixed_tw.items():
    s = summarize_errors(arr)
    paper_rows.append({
        "group": "fixed_eps_twinkle_auditor",
        "method": name,
        **s
    })

# 5) eps-scaled
for name, arr in agg_scaled.items():
    s = summarize_errors(arr)
    paper_rows.append({
        "group": "eps_scaled",
        "method": name,
        **s
    })

paper_stats_df = pd.DataFrame(paper_rows)
paper_stats_csv = os.path.join(OUT_DIR, "exp3_geolife_paper_stats.csv")
paper_stats_df.to_csv(paper_stats_csv, index=False)

# also save json
paper_stats_json = os.path.join(OUT_DIR, "exp3_geolife_paper_stats.json")
with open(paper_stats_json, "w", encoding="utf-8") as f:
    json.dump(paper_rows, f, ensure_ascii=False, indent=2)

# per-trajectory meta
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(os.path.join(OUT_DIR, "exp3_geolife_meta.csv"), index=False)

# %%
# ============================
# Console summary
# ============================

print("\n=== Paper-ready stats ===")
print(paper_stats_df)

elapsed = time.time() - start_time
print(f"\nDone. Total elapsed: {elapsed:.1f}s")
print(f"Figures saved to: {OUT_DIR}")
print(f"Paper stats CSV: {paper_stats_csv}")
print(f"Paper stats JSON: {paper_stats_json}")