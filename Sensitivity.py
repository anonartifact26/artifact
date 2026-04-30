#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional


# =========================
# Matplotlib configuration
# =========================

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 320
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Liberation Sans",
    "Noto Sans",
]
plt.rcParams["axes.facecolor"] = "#FAFAFA"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["grid.color"] = "#D9D9D9"
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 0.92
plt.rcParams["legend.fancybox"] = True


# =========================
# Plotting constants
# =========================

COLOR_PRIMARY = "#3366CC"
COLOR_SECOND = "#DC3912"
COLOR_THIRD = "#FF9900"
COLOR_FOURTH = "#109618"
COLOR_FIFTH = "#990099"
COLOR_SIXTH = "#0099C6"

RISK_COLORS = {
    "mean_r_route_mean": "#4C78A8",
    "mean_r_place_mean": "#F58518",
}

LABEL_MAP = {
    "budget_B": "budget_B",
    "risk_tau": "risk_tau",

    "err_med_all_mean": "Median error (all emissions)",
    "err_p95_all_mean": "P95 error (all emissions)",
    "err_mean_all_mean": "Mean error (all emissions)",
    "err_med_burst_mean": "Median error (burst windows)",
    "err_p95_burst_mean": "P95 error (burst windows)",
    "err_mean_burst_mean": "Mean error (burst windows)",

    "mean_R_total_mean": "Composite risk",
    "mean_r_route_mean": "Route risk",
    "mean_r_place_mean": "Place risk",

    "emit_rate_mean": "Emission rate",
    "budget_final_mean": "Final budget consumed",
    "avg_alpha_mean": "Average alpha",
    "avg_eps_eff_mean": "Average effective epsilon",
}


# ============================================================
# WGS84 -> ECEF -> ENU
# ============================================================

a = 6378137.0
f = 1 / 298.257223563
e2 = 2 * f - f * f


def wgs84_to_ecef(lat_deg, lon_deg, h_m):
    """
    Convert WGS84 coordinates to Earth-Centered, Earth-Fixed (ECEF) coordinates.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    X = (N + h_m) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h_m) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + h_m) * np.sin(lat_rad)
    return X, Y, Z


def ecef_to_enu(X, Y, Z, lat0_deg, lon0_deg, h0_m):
    """
    Convert ECEF coordinates to a local East-North-Up (ENU) frame.
    """
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
        [-sin_lat * cos_lon,  -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon,    cos_lat * sin_lon, sin_lat],
    ])

    enu = t @ np.vstack((dx, dy, dz))
    return enu[0], enu[1], enu[2]


# ============================================================
# Capability-tier simulation
# Main paper results use the L1-only setting
# ============================================================

SCALE_L1_ONLY = 4.0
NLOS_BIAS = np.array([0.8, -1.5])


def simulate_internal_estimate(
    X_t,
    capability: str = "L1-only",
    add_bias: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate an internal location estimate under the reported device capability tier.
    """
    rng = np.random.default_rng(seed)
    T, D = X_t.shape

    if capability != "L1-only":
        raise ValueError("This script is restricted to the L1-only setting reported in the paper.")

    noise_scale = SCALE_L1_ONLY
    noise = rng.laplace(loc=0.0, scale=noise_scale, size=(T, D))
    bias = NLOS_BIAS if add_bias else 0.0
    return X_t + noise + bias


# ============================================================
# Planar Laplace mechanism
# ============================================================

def planar_laplace_noise(eps: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample planar Laplace noise for n two-dimensional points.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")

    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    r = rng.gamma(shape=2.0, scale=1.0 / eps, size=n)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return np.column_stack([dx, dy])


def planar_laplace_release(x: np.ndarray, eps_eff: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply the planar Laplace mechanism to a single point or an array of 2D points.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        noise = planar_laplace_noise(eps_eff, 1, rng)[0]
        return x + noise

    noise = planar_laplace_noise(eps_eff, x.shape[0], rng)
    return x + noise


# ============================================================
# Reference burst schedule
# Three synthetic intents are placed at relative positions
# 25%, 55%, and 80%, each with duration 60 s
# ============================================================

def make_reference_burst_mask(
    T: int,
    dt: float,
    burst_len_s: float = 60.0,
    starts: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Construct the fixed reference burst mask used throughout the paper.
    """
    L = max(1, int(round(burst_len_s / dt)))
    mask = np.zeros(T, dtype=bool)

    if starts is None:
        starts = [int(0.25 * T), int(0.55 * T), int(0.80 * T)]

    for s in starts:
        s = max(0, min(T - 1, s))
        e = min(T, s + L)
        mask[s:e] = True

    return mask


def should_emit(t: int, last_emit_t: Optional[int], interval_steps: int) -> bool:
    """
    Determine whether the mechanism is allowed to emit at time step t.
    """
    if last_emit_t is None:
        return True
    return (t - last_emit_t) >= interval_steps


# ============================================================
# Route/place auditor
# The implemented risk combines route and place components only,
# matching the formulation described in the paper
# ============================================================

@dataclass
class AuditorParams:
    vmax_mps: float = 15.0
    state_stride: int = 20
    transition_slack: float = 1.25

    place_hist_s: float = 120.0

    w_route: float = 0.5
    w_place: float = 0.5

    risk_tau: float = 0.6
    gamma: float = 5.0
    alpha_max: float = 25.0


class TwinkleAuditor:
    """
    Causal privacy auditor based solely on sanitized history.

    Route risk proxy:
        Posterior concentration over a downsampled set of reference states.

    Place risk proxy:
        Concentration toward designated sensitive points of interest inferred
        from the sanitized release history.

    Composite risk:
        Weighted combination of route and place risk, consistent with the
        formulation used in the paper.
    """

    def __init__(
        self,
        x_reference: np.ndarray,
        dt: float,
        geofence_centers: List[np.ndarray],
        params: AuditorParams,
        rng: np.random.Generator,
    ):
        self.dt = dt
        self.p = params
        self.rng = rng

        self.S = x_reference[::self.p.state_stride].copy()
        self.Ns = self.S.shape[0]

        self.belief = np.ones(self.Ns, dtype=float) / self.Ns
        self.last_release_t: Optional[int] = None

        self.pois = geofence_centers[:] if len(geofence_centers) > 0 else [x_reference[0]]

        self.released_times: List[int] = []
        self.released_points: List[np.ndarray] = []
        self.released_eps: List[float] = []

        self.dmat = np.linalg.norm(self.S[:, None, :] - self.S[None, :, :], axis=2)

    def _route_risk_from_release(self, z_t: np.ndarray, eps_eff: float, t: int) -> float:
        if self.last_release_t is None:
            delta_t = self.dt
        else:
            delta_t = (t - self.last_release_t) * self.dt

        thr = self.p.vmax_mps * delta_t * self.p.transition_slack

        prev = self.belief
        next_prior = np.zeros_like(prev)
        feasible = self.dmat <= thr

        for i in range(self.Ns):
            js = np.where(feasible[i])[0]
            if js.size == 0:
                next_prior[i] += prev[i]
            else:
                next_prior[js] += prev[i] / js.size

        dist = np.linalg.norm(self.S - z_t[None, :], axis=1)
        emit_like = np.exp(-eps_eff * dist)

        post = next_prior * emit_like
        s = post.sum()

        if s <= 0 or (not np.isfinite(s)):
            self.belief = np.ones(self.Ns, dtype=float) / self.Ns
        else:
            self.belief = post / s

        self.last_release_t = t
        return float(np.clip(self.belief.max(), 0.0, 1.0))

    def _route_risk_without_release(self) -> float:
        if self.belief.size == 0:
            return 0.0
        return float(np.clip(self.belief.max(), 0.0, 1.0))

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

        weights = np.asarray(weights, dtype=float).reshape(-1)
        s = weights.sum()
        if s <= 0 or (not np.isfinite(s)):
            return 0.0

        post = weights / s
        sens_idx = 0
        return float(np.clip(post[sens_idx], 0.0, 1.0))

    def compute_alpha_for_candidate(
        self,
        t: int,
        z_prov: Optional[np.ndarray],
        eps_nom_t: float,
    ) -> Tuple[float, float, float, float]:
        """
        Score a provisional sanitized candidate using only sanitized history,
        then derive the alpha_t inflation factor.

        Returns
        -------
        Tuple[float, float, float, float]
            Route risk, place risk, composite risk, and alpha_t.
        """
        if z_prov is None:
            r_route = self._route_risk_without_release()
            r_place = self._place_risk_proxy(t)
        else:
            # Evaluate the candidate without committing it to persistent state.
            belief_backup = self.belief.copy()
            last_release_backup = self.last_release_t

            r_route = self._route_risk_from_release(z_prov, eps_nom_t, t)

            self.belief = belief_backup
            self.last_release_t = last_release_backup

            r_place = self._place_risk_proxy(t)

        wsum = self.p.w_route + self.p.w_place
        R = (self.p.w_route * r_route + self.p.w_place * r_place) / wsum
        R = float(np.clip(R, 0.0, 1.0))

        alpha = 1.0 + self.p.gamma * max(0.0, R - self.p.risk_tau)
        alpha = float(np.clip(alpha, 1.0, self.p.alpha_max))

        return float(r_route), float(r_place), float(R), float(alpha)

    def commit_release(self, t: int, z_t: np.ndarray, eps_eff: float) -> Tuple[float, float, float]:
        """
        Commit an accepted release to the auditor state and return updated risks.
        """
        r_route = self._route_risk_from_release(z_t, eps_eff, t)
        self.released_times.append(t)
        self.released_points.append(z_t.copy())
        self.released_eps.append(float(eps_eff))
        r_place = self._place_risk_proxy(t)

        wsum = self.p.w_route + self.p.w_place
        R = (self.p.w_route * r_route + self.p.w_place * r_place) / wsum
        R = float(np.clip(R, 0.0, 1.0))
        return float(r_route), float(r_place), float(R)

    def no_release_step(self, t: int) -> Tuple[float, float, float]:
        """
        Update risk summaries for a time step at which no release is emitted.
        """
        r_route = self._route_risk_without_release()
        r_place = self._place_risk_proxy(t)

        wsum = self.p.w_route + self.p.w_place
        R = (self.p.w_route * r_route + self.p.w_place * r_place) / wsum
        R = float(np.clip(R, 0.0, 1.0))
        return float(r_route), float(r_place), float(R)


# ============================================================
# Twinkle parameters
# Default values are aligned with the paper configuration
# ============================================================

@dataclass
class TwinkleParams:
    dt: float = 1.0
    baseline_interval_s: float = 30.0
    burst_interval_s: float = 2.0
    burst_len_s: float = 60.0

    eps_base: float = 0.03
    eps_burst: float = 0.12
    budget_B: float = 6.0

    geofence_radius_m: float = 30.0
    geofence_indices: Tuple[int, ...] = (120, 260)

    use_auditor: bool = True
    auditor_params: AuditorParams = field(default_factory=AuditorParams)

    release_center: str = "xhat"


# ============================================================
# Twinkle simulator
# Burst metrics are evaluated against the fixed reference burst
# schedule rather than adaptive release times
# ============================================================

def simulate_twinkle(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    params: TwinkleParams,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate the Twinkle mechanism on a single trajectory.
    """
    rng = np.random.default_rng(seed)
    T = x_true.shape[0]
    dt = params.dt

    ref_burst = make_reference_burst_mask(T, dt, params.burst_len_s)

    base_steps = max(1, int(round(params.baseline_interval_s / dt)))
    burst_steps = max(1, int(round(params.burst_interval_s / dt)))

    poi_centers = []
    for idx in params.geofence_indices:
        idx = int(np.clip(idx, 0, T - 1))
        poi_centers.append(x_true[idx].copy())

    auditor = None
    if params.use_auditor:
        auditor = TwinkleAuditor(
            x_reference=x_true,
            dt=dt,
            geofence_centers=poi_centers if len(poi_centers) > 0 else [x_true[0]],
            params=params.auditor_params,
            rng=rng,
        )

    z = np.full((T, 2), np.nan, dtype=float)
    a_emit = np.zeros(T, dtype=int)
    m = ref_burst.astype(int)

    alpha = np.ones(T, dtype=float)
    eps_nom = np.zeros(T, dtype=float)
    eps_eff = np.zeros(T, dtype=float)
    E_budget = np.zeros(T, dtype=float)

    r_route = np.zeros(T, dtype=float)
    r_place = np.zeros(T, dtype=float)
    R_total = np.zeros(T, dtype=float)

    last_emit_t: Optional[int] = None
    budget_used = 0.0

    x_center = x_true if params.release_center == "truth" else x_hat

    for t in range(T):
        eps_nom_t = (1 - m[t]) * params.eps_base + m[t] * params.eps_burst
        eps_nom[t] = eps_nom_t

        interval_steps = burst_steps if m[t] == 1 else base_steps
        cadence_ok = should_emit(t, last_emit_t, interval_steps)

        emit = cadence_ok
        alpha_t = 1.0

        if emit and auditor is not None:
            # Provisional release used only for candidate scoring.
            z_prov = planar_laplace_release(x_center[t], eps_nom_t, rng)
            rr, rp, R, alpha_t = auditor.compute_alpha_for_candidate(
                t=t,
                z_prov=z_prov,
                eps_nom_t=eps_nom_t,
            )
        elif auditor is not None:
            rr, rp, R, _ = auditor.compute_alpha_for_candidate(
                t=t,
                z_prov=None,
                eps_nom_t=eps_nom_t,
            )
        else:
            rr, rp, R = 0.0, 0.0, 0.0

        alpha[t] = alpha_t
        eps_eff_t = eps_nom_t / alpha_t
        eps_eff[t] = eps_eff_t

        remaining = params.budget_B - budget_used

        # Hard budget enforcement: increase alpha when possible; otherwise suppress.
        if emit and remaining < eps_eff_t:
            if remaining > 0:
                alpha_needed = eps_nom_t / remaining
                alpha_cap = params.auditor_params.alpha_max if params.use_auditor else 1e9
                alpha_new = float(np.clip(max(alpha_t, alpha_needed), 1.0, alpha_cap))
                eps_eff_t2 = eps_nom_t / alpha_new

                if remaining >= eps_eff_t2:
                    alpha_t = alpha_new
                    eps_eff_t = eps_eff_t2
                    alpha[t] = alpha_t
                    eps_eff[t] = eps_eff_t
                else:
                    emit = False
            else:
                emit = False

        if emit:
            z_t = planar_laplace_release(x_center[t], eps_eff_t, rng)
            z[t] = z_t
            a_emit[t] = 1
            last_emit_t = t
            budget_used += eps_eff_t

            if auditor is not None:
                rr, rp, R = auditor.commit_release(t=t, z_t=z_t, eps_eff=eps_eff_t)
        else:
            a_emit[t] = 0
            if auditor is not None:
                rr, rp, R = auditor.no_release_step(t=t)

        r_route[t] = rr
        r_place[t] = rp
        R_total[t] = R
        E_budget[t] = budget_used

    return pd.DataFrame({
        "t": np.arange(T) * dt,
        "xE": x_true[:, 0], "xN": x_true[:, 1],
        "xhatE": x_hat[:, 0], "xhatN": x_hat[:, 1],
        "zE": z[:, 0], "zN": z[:, 1],
        "a": a_emit,
        "mode": m,
        "ref_burst": ref_burst.astype(int),
        "alpha": alpha,
        "eps_nom": eps_nom,
        "eps_eff": eps_eff,
        "budget_used": E_budget,
        "r_route": r_route,
        "r_place": r_place,
        "R_total": R_total,
    })


# ============================================================
# Metrics
# ============================================================

def emission_errors(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Euclidean errors for all emitted releases.
    """
    dfe = df[df["a"] == 1].copy()
    if dfe.empty:
        return np.array([])

    z = dfe[["zE", "zN"]].to_numpy()
    x = dfe[["xE", "xN"]].to_numpy()
    return np.linalg.norm(z - x, axis=1)


def emission_errors_burst_only(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Euclidean errors restricted to emissions that occur within
    the fixed reference burst windows.
    """
    if "ref_burst" not in df.columns:
        return np.array([])

    dfe = df[(df["a"] == 1) & (df["ref_burst"] == 1)].copy()
    if dfe.empty:
        return np.array([])

    z = dfe[["zE", "zN"]].to_numpy()
    x = dfe[["xE", "xN"]].to_numpy()
    return np.linalg.norm(z - x, axis=1)


def summarize_df_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize release behavior, budget usage, risk, and utility metrics
    for a single simulated trajectory.
    """
    errs_all = emission_errors(df)
    errs_burst = emission_errors_burst_only(df)

    emit_mask = df["a"] == 1
    n_emit = int(emit_mask.sum())
    n_burst_emit = int(((df["a"] == 1) & (df["ref_burst"] == 1)).sum())
    T = len(df)

    return {
        "n_steps": T,
        "n_emit": n_emit,
        "n_burst_emit": n_burst_emit,
        "emit_rate": float(n_emit / T) if T > 0 else 0.0,
        "budget_final": float(df["budget_used"].iloc[-1]) if T > 0 else 0.0,
        "avg_alpha": float(df["alpha"].mean()) if "alpha" in df.columns else np.nan,
        "avg_eps_eff": float(df["eps_eff"].mean()) if "eps_eff" in df.columns else np.nan,
        "mean_R_total": float(df["R_total"].mean()) if "R_total" in df.columns else np.nan,
        "mean_r_route": float(df["r_route"].mean()) if "r_route" in df.columns else np.nan,
        "mean_r_place": float(df["r_place"].mean()) if "r_place" in df.columns else np.nan,
        "err_med_all": float(np.median(errs_all)) if errs_all.size > 0 else np.nan,
        "err_p95_all": float(np.quantile(errs_all, 0.95)) if errs_all.size > 0 else np.nan,
        "err_mean_all": float(np.mean(errs_all)) if errs_all.size > 0 else np.nan,
        "err_med_burst": float(np.median(errs_burst)) if errs_burst.size > 0 else np.nan,
        "err_p95_burst": float(np.quantile(errs_burst, 0.95)) if errs_burst.size > 0 else np.nan,
        "err_mean_burst": float(np.mean(errs_burst)) if errs_burst.size > 0 else np.nan,
    }


# ============================================================
# GeoLife loading and preprocessing
# The preprocessing follows the paper setup:
# - retain trajectories with at least 80 samples
# - truncate retained trajectories to at most 1200 samples
# - keep at most three trajectories per user
# ============================================================

def load_geolife_plt(path: str) -> pd.DataFrame:
    """
    Load a GeoLife trajectory file in .plt format.
    """
    df = pd.read_csv(path, skiprows=6, header=None)
    df.columns = ["lat", "lon", "zero", "alt", "days", "date", "time"]
    return df


def geolife_df_to_local_xy(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a GeoLife dataframe to a local ENU XY trajectory.
    """
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    h = df["alt"].to_numpy()
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    X, Y, Z = wgs84_to_ecef(lat, lon, h)
    lat0, lon0, h0 = lat[0], lon[0], h[0]
    E, N, _ = ecef_to_enu(X, Y, Z, lat0, lon0, h0)
    return np.column_stack([E, N])


def collect_geolife_trajectories(
    root: str,
    min_len: int = 80,
    max_len: int = 1200,
    max_users: Optional[int] = None,
    max_traj_per_user: int = 3,
) -> List[np.ndarray]:
    """
    Collect valid GeoLife trajectories subject to the preprocessing rules
    used in the paper.
    """
    user_dirs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
    if max_users is not None:
        user_dirs = user_dirs[:max_users]

    print(f"[INFO] Found {len(user_dirs)} user directories under: {root}")

    trajs = []
    for ui, user_dir in enumerate(user_dirs):
        traj_dir = os.path.join(user_dir, "Trajectory")
        print(f"[INFO] Scanning user {ui + 1}/{len(user_dirs)}: {os.path.basename(user_dir)}")

        if not os.path.isdir(traj_dir):
            print(f"[WARN] Missing trajectory directory: {traj_dir}")
            continue

        files = sorted(glob.glob(os.path.join(traj_dir, "*.plt")))
        print(f"[INFO]   Found {len(files)} trajectory files")

        accepted = 0
        for fi, fp in enumerate(files):
            if accepted >= max_traj_per_user:
                break

            try:
                if fi == 0 or (fi + 1) % 10 == 0:
                    print(f"[INFO]   Reading file {fi + 1}/{len(files)}")

                df = load_geolife_plt(fp)
                if len(df) < min_len:
                    continue

                df = df.iloc[:max_len].copy()
                x_true_local = geolife_df_to_local_xy(df)

                span = np.linalg.norm(
                    np.nanmax(x_true_local, axis=0) - np.nanmin(x_true_local, axis=0)
                )
                if not np.isfinite(span) or span < 30.0:
                    continue

                trajs.append(x_true_local)
                accepted += 1

            except Exception as e:
                print(f"[WARN] Failed to load {fp}: {e}")

        print(f"[INFO]   Accepted {accepted} valid trajectories from this user")

    print(f"[INFO] Loaded {len(trajs)} valid GeoLife trajectories in total")
    return trajs


# ============================================================
# Plot helpers
# Figure styles and filename patterns are kept consistent with
# the rest of the artifact
# ============================================================

def ensure_dir(path: str):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def save_results_table(df: pd.DataFrame, save_path: str):
    """
    Save a dataframe as a UTF-8 encoded CSV file.
    """
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved CSV: {save_path}")


def pretty_label(name: str) -> str:
    """
    Map an internal metric name to a presentation-friendly label.
    """
    return LABEL_MAP.get(name, name)


def normalize_series(y: np.ndarray) -> np.ndarray:
    """
    Normalize a numeric array to the range [0, 1] while preserving NaNs.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y

    finite = np.isfinite(y)
    if finite.sum() == 0:
        return np.full_like(y, np.nan, dtype=float)

    ymin = np.nanmin(y)
    ymax = np.nanmax(y)

    if np.isclose(ymax, ymin):
        out = np.zeros_like(y, dtype=float)
        out[finite] = 0.5
        return out

    return (y - ymin) / (ymax - ymin)


def plot_utility_ribbon(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot utility metrics across the sensitivity sweep using line-and-ribbon style.
    """
    x = df[x_col].to_numpy()

    metrics = [
        ("err_med_all_mean", None, COLOR_PRIMARY, "Median error (all emissions)"),
        ("err_p95_all_mean", None, COLOR_SECOND, "P95 error (all emissions)"),
        ("err_med_burst_mean", None, COLOR_FOURTH, "Median error (burst windows)"),
        ("err_p95_burst_mean", None, COLOR_THIRD, "P95 error (burst windows)"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    any_line = False

    for col, _, color, label in metrics:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() == 0:
            continue

        any_line = True
        ax.plot(x[ok], y[ok], color=color, marker="o", linewidth=2.4, markersize=6, label=label)
        ax.fill_between(x[ok], y[ok] * 0.95, y[ok] * 1.05, color=color, alpha=0.12)

    if not any_line:
        plt.close(fig)
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    ax.set_title(f"Utility sensitivity overview - {pretty_label(x_col)} ({capability})")
    ax.set_xlabel(pretty_label(x_col))
    ax.set_ylabel("Error metric")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


def plot_risk_area(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot stacked route/place risk composition across the sensitivity sweep.
    """
    x = df[x_col].to_numpy(dtype=float)
    cols = ["mean_r_route_mean", "mean_r_place_mean"]

    ys = []
    labels = []
    colors = []

    for c in cols:
        if c in df.columns:
            y = df[c].to_numpy(dtype=float)
            ys.append(np.nan_to_num(y, nan=0.0))
            labels.append(pretty_label(c))
            colors.append(RISK_COLORS[c])

    if len(ys) == 0:
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.stackplot(x, ys, labels=labels, colors=colors, alpha=0.85)

    if "mean_R_total_mean" in df.columns:
        y_total = df["mean_R_total_mean"].to_numpy(dtype=float)
        ax.plot(x, y_total, color="#222222", linewidth=2.5, marker="o", label="Composite risk")

    ax.set_title(f"Privacy-risk composition - {pretty_label(x_col)} ({capability})")
    ax.set_xlabel(pretty_label(x_col))
    ax.set_ylabel("Risk value")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


def plot_behavior_bar(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot normalized mechanism-behavior metrics as grouped bars.
    """
    x_raw = df[x_col].to_numpy()
    x_labels = [str(v) for v in x_raw]
    idx = np.arange(len(x_labels))

    metrics = [
        ("emit_rate_mean", COLOR_PRIMARY, "Emission rate"),
        ("budget_final_mean", COLOR_SECOND, "Final budget consumed"),
        ("avg_alpha_mean", COLOR_FOURTH, "Average alpha"),
        ("avg_eps_eff_mean", COLOR_THIRD, "Average effective epsilon"),
    ]

    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6.4))

    any_bar = False
    for i, (col, color, label) in enumerate(metrics):
        if col not in df.columns:
            continue
        y = normalize_series(df[col].to_numpy(dtype=float))
        if np.isfinite(y).sum() == 0:
            continue

        any_bar = True
        ax.bar(idx + (i - 1.5) * width, y, width=width, color=color, alpha=0.88, label=label)

    if not any_bar:
        plt.close(fig)
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    ax.set_xticks(idx)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(pretty_label(x_col))
    ax.set_ylabel("Normalized value")
    ax.set_title(f"Mechanism behavior comparison - {pretty_label(x_col)} ({capability})")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


def plot_tradeoff_bubble(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot a utility-risk trade-off view using bubble size and color to encode
    additional mechanism behavior statistics.
    """
    if (
        "err_med_all_mean" not in df.columns or
        "mean_R_total_mean" not in df.columns or
        "budget_final_mean" not in df.columns or
        "emit_rate_mean" not in df.columns
    ):
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    x = df["err_med_all_mean"].to_numpy(dtype=float)
    y = df["mean_R_total_mean"].to_numpy(dtype=float)
    size_base = df["budget_final_mean"].to_numpy(dtype=float)
    color_val = df["emit_rate_mean"].to_numpy(dtype=float)
    param_val = df[x_col].to_numpy()

    if np.isfinite(x).sum() == 0 or np.isfinite(y).sum() == 0:
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    s = 200 + 900 * normalize_series(size_base)

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    sc = ax.scatter(
        x, y, s=s, c=color_val, cmap="viridis", alpha=0.85,
        edgecolors="black", linewidths=0.8
    )

    for xi, yi, pv in zip(x, y, param_val):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(pv), (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Emission rate")

    ax.set_xlabel("Median error (all emissions)")
    ax.set_ylabel("Composite risk")
    ax.set_title(f"Trade-off bubble view - {pretty_label(x_col)} ({capability})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


def plot_metric_heatmap(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot a normalized metric heatmap across the sensitivity sweep.
    """
    metric_cols = [
        "err_med_all_mean",
        "err_p95_all_mean",
        "err_med_burst_mean",
        "err_p95_burst_mean",
        "mean_R_total_mean",
        "emit_rate_mean",
        "budget_final_mean",
        "avg_alpha_mean",
        "avg_eps_eff_mean",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    if len(cols) == 0:
        print(f"[WARN] Skipping empty plot: {save_path}")
        return

    matrix = []
    y_labels = []
    for c in cols:
        matrix.append(normalize_series(df[c].to_numpy(dtype=float)))
        y_labels.append(pretty_label(c))
    M = np.vstack(matrix)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", interpolation="nearest")

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([str(v) for v in df[x_col].to_list()])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(pretty_label(x_col))
    ax.set_title(f"Normalized metric heatmap - {pretty_label(x_col)} ({capability})")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt = "nan" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="#1A1A1A")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized value")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


def plot_summary_dashboard(df: pd.DataFrame, x_col: str, save_path: str, capability: str):
    """
    Plot a multi-panel dashboard summarizing utility, risk, behavior,
    and trade-off trends.
    """
    x = df[x_col].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    for col, color, label in [
        ("err_med_all_mean", COLOR_PRIMARY, "Median error (all emissions)"),
        ("err_p95_all_mean", COLOR_SECOND, "P95 error (all emissions)"),
        ("err_med_burst_mean", COLOR_FOURTH, "Median error (burst windows)"),
    ]:
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 0:
                ax1.plot(x[ok], y[ok], marker="o", linewidth=2.2, label=label, color=color)

    ax1.set_title("Utility")
    ax1.set_xlabel(pretty_label(x_col))
    ax1.set_ylabel("Error")
    ax1.legend(fontsize=8)

    risk_cols = [c for c in ["mean_r_route_mean", "mean_r_place_mean"] if c in df.columns]
    if len(risk_cols) > 0:
        ys = [np.nan_to_num(df[c].to_numpy(dtype=float), nan=0.0) for c in risk_cols]
        labels = [pretty_label(c) for c in risk_cols]
        colors = [RISK_COLORS[c] for c in risk_cols]
        ax2.stackplot(x, ys, labels=labels, colors=colors, alpha=0.85)

    if "mean_R_total_mean" in df.columns:
        y = df["mean_R_total_mean"].to_numpy(dtype=float)
        ax2.plot(x, y, color="#111111", linewidth=2.2, marker="o", label="Composite risk")

    ax2.set_title("Privacy risk")
    ax2.set_xlabel(pretty_label(x_col))
    ax2.set_ylabel("Risk")
    ax2.legend(fontsize=8)

    for col, color, label in [
        ("emit_rate_mean", COLOR_PRIMARY, "Emission rate"),
        ("budget_final_mean", COLOR_SECOND, "Final budget consumed"),
        ("avg_alpha_mean", COLOR_FOURTH, "Average alpha"),
    ]:
        if col in df.columns:
            y = df[col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 0:
                ax3.plot(x[ok], y[ok], marker="s", linewidth=2.2, label=label, color=color)

    ax3.set_title("Mechanism behavior")
    ax3.set_xlabel(pretty_label(x_col))
    ax3.set_ylabel("Value")
    ax3.legend(fontsize=8)

    if "err_med_all_mean" in df.columns and "mean_R_total_mean" in df.columns:
        xs = df["err_med_all_mean"].to_numpy(dtype=float)
        ys = df["mean_R_total_mean"].to_numpy(dtype=float)
        cs = df["emit_rate_mean"].to_numpy(dtype=float) if "emit_rate_mean" in df.columns else np.zeros_like(xs)
        ss = (
            120 + 600 * normalize_series(df["budget_final_mean"].to_numpy(dtype=float))
            if "budget_final_mean" in df.columns else 180
        )
        sc = ax4.scatter(xs, ys, c=cs, s=ss, cmap="viridis", edgecolors="black", linewidths=0.7, alpha=0.85)

        for xi, yi, pv in zip(xs, ys, df[x_col].to_numpy()):
            if np.isfinite(xi) and np.isfinite(yi):
                ax4.annotate(str(pv), (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

        fig.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)

    ax4.set_title("Trade-off map")
    ax4.set_xlabel("Median error (all emissions)")
    ax4.set_ylabel("Composite risk")

    fig.suptitle(f"Sensitivity dashboard - {pretty_label(x_col)} ({capability})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {save_path}")


# ============================================================
# Single-trajectory evaluation under one parameter setting
# Default values are aligned with the paper configuration
# ============================================================

def build_default_twinkle_params(T: int, dt: float = 1.0) -> "TwinkleParams":
    """
    Build the default Twinkle parameter set for a trajectory of length T.
    """
    g1 = int(np.clip(0.30 * T, 0, T - 1))
    g2 = int(np.clip(0.70 * T, 0, T - 1))

    return TwinkleParams(
        dt=dt,
        baseline_interval_s=30.0,
        burst_interval_s=2.0,
        burst_len_s=60.0,
        eps_base=0.03,
        eps_burst=0.12,
        budget_B=6.0,
        geofence_radius_m=30.0,
        geofence_indices=(g1, g2),
        use_auditor=True,
        auditor_params=AuditorParams(
            vmax_mps=15.0,
            state_stride=20,
            transition_slack=1.25,
            place_hist_s=120.0,
            w_route=0.5,
            w_place=0.5,
            risk_tau=0.6,
            gamma=5.0,
            alpha_max=25.0,
        ),
        release_center="xhat",
    )


def run_twinkle_on_one_traj_with_param(
    x_true: np.ndarray,
    traj_seed: int,
    param_name: str,
    param_value: float,
    dt: float = 1.0,
    capability: str = "L1-only",
) -> pd.DataFrame:
    """
    Run the Twinkle mechanism on a single trajectory with one parameter override.
    """
    x_hat = simulate_internal_estimate(x_true, capability=capability, seed=traj_seed + 1)
    base_params = build_default_twinkle_params(len(x_true), dt=dt)

    if param_name == "budget_B":
        params = replace(base_params, budget_B=float(param_value))
    elif param_name == "risk_tau":
        new_auditor = replace(base_params.auditor_params, risk_tau=float(param_value))
        params = replace(base_params, auditor_params=new_auditor)
    else:
        raise ValueError(f"Unsupported param_name: {param_name}")

    return simulate_twinkle(x_true, x_hat, params, seed=traj_seed + 10)


# ============================================================
# Sensitivity experiment
# Only budget_B and risk_tau are swept, consistent with the paper
# ============================================================

def run_sensitivity_experiment(
    geolife_root: str,
    out_dir: str = "sensitivity_results",
    max_users: Optional[int] = None,
    min_len: int = 80,
    max_len: int = 1200,
    capability: str = "L1-only",
    budget_grid: Optional[List[float]] = None,
    risk_tau_grid: Optional[List[float]] = None,
):
    """
    Run the sensitivity study over the specified parameter grids.
    """
    if capability != "L1-only":
        raise ValueError("This script is restricted to the L1-only setting reported in the paper.")

    if budget_grid is None:
        budget_grid = [2.0, 4.0, 6.0, 8.0, 10.0]
    if risk_tau_grid is None:
        risk_tau_grid = [0.4, 0.5, 0.6, 0.7, 0.8]

    ensure_dir(out_dir)

    print("[INFO] Loading trajectories...")
    trajectories = collect_geolife_trajectories(
        root=geolife_root,
        min_len=min_len,
        max_len=max_len,
        max_users=max_users,
        max_traj_per_user=3,
    )

    if len(trajectories) == 0:
        raise RuntimeError("No valid GeoLife trajectories found.")

    print(f"[INFO] Loaded {len(trajectories)} trajectories for sensitivity analysis")

    experiments = {
        "budget_B": budget_grid,
        "risk_tau": risk_tau_grid,
    }

    all_outputs = {}

    for exp_name, grid in experiments.items():
        print(f"\n[INFO] ===== Sensitivity sweep for {exp_name} =====")
        rows = []

        for gv, val in enumerate(grid):
            t0 = time.time()
            print(f"[INFO] {exp_name} = {val} ({gv + 1}/{len(grid)})")

            metrics_list = []
            for i, x_true_i in enumerate(trajectories):
                df = run_twinkle_on_one_traj_with_param(
                    x_true=x_true_i,
                    traj_seed=1000 + i * 100,
                    param_name=exp_name,
                    param_value=val,
                    dt=1.0,
                    capability=capability,
                )
                metrics_list.append(summarize_df_metrics(df))

            md = pd.DataFrame(metrics_list)

            row = {
                exp_name: val,
                "n_traj": len(metrics_list),

                "emit_rate_mean": md["emit_rate"].mean(),
                "emit_rate_std": md["emit_rate"].std(ddof=0),

                "budget_final_mean": md["budget_final"].mean(),
                "budget_final_std": md["budget_final"].std(ddof=0),

                "avg_alpha_mean": md["avg_alpha"].mean(),
                "avg_alpha_std": md["avg_alpha"].std(ddof=0),

                "avg_eps_eff_mean": md["avg_eps_eff"].mean(),
                "avg_eps_eff_std": md["avg_eps_eff"].std(ddof=0),

                "mean_R_total_mean": md["mean_R_total"].mean(),
                "mean_R_total_std": md["mean_R_total"].std(ddof=0),

                "mean_r_route_mean": md["mean_r_route"].mean(),
                "mean_r_place_mean": md["mean_r_place"].mean(),

                "err_med_all_mean": md["err_med_all"].mean(),
                "err_p95_all_mean": md["err_p95_all"].mean(),
                "err_mean_all_mean": md["err_mean_all"].mean(),

                "err_med_burst_mean": md["err_med_burst"].mean(),
                "err_p95_burst_mean": md["err_p95_burst"].mean(),
                "err_mean_burst_mean": md["err_mean_burst"].mean(),

                "n_burst_emit_mean": md["n_burst_emit"].mean(),
            }
            rows.append(row)

            elapsed = time.time() - t0
            print(
                f"[INFO] Completed {exp_name}={val} in {elapsed:.2f}s | "
                f"err_med_all={row['err_med_all_mean']:.2f}, "
                f"err_med_burst={row['err_med_burst_mean']:.2f}, "
                f"R={row['mean_R_total_mean']:.3f}, "
                f"emit_rate={row['emit_rate_mean']:.3f}"
            )

        result_df = pd.DataFrame(rows).sort_values(by=exp_name).reset_index(drop=True)
        all_outputs[exp_name] = result_df

        csv_path = os.path.join(out_dir, f"sensitivity_{exp_name}.csv")
        save_results_table(result_df, csv_path)

        plot_summary_dashboard(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_summary.png"),
            capability,
        )

        plot_utility_ribbon(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_utility_ribbon.png"),
            capability,
        )

        plot_risk_area(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_risk_area.png"),
            capability,
        )

        plot_behavior_bar(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_behavior_bar.png"),
            capability,
        )

        plot_tradeoff_bubble(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_tradeoff_bubble.png"),
            capability,
        )

        plot_metric_heatmap(
            result_df,
            exp_name,
            os.path.join(out_dir, f"sensitivity_{exp_name}_heatmap.png"),
            capability,
        )

    print("\n[INFO] ===== Sensitivity experiment finished =====")
    for name, df in all_outputs.items():
        print(f"\n[INFO] Result summary for {name}:")
        print(df.to_string(index=False))

    return all_outputs


# ============================================================
# CLI entry point
# ============================================================

def main():
    geolife_root = "path/to/Geolife/Data"
    out_dir = "sensitivity_results_paper_consistent"

    # For the sensitivity study, the paper uses a larger pool of valid
    # trajectories. With max_traj_per_user fixed at three, setting
    # max_users=None allows the full dataset to be scanned.
    max_users = None

    min_len = 80
    max_len = 1200
    capability = "L1-only"

    print("[INFO] Sensitivity experiment started")
    print(f"[INFO] geolife_root = {geolife_root}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] max_users = {max_users}")
    print(f"[INFO] min_len = {min_len}")
    print(f"[INFO] max_len = {max_len}")
    print(f"[INFO] capability = {capability}")

    if not os.path.isdir(geolife_root):
        raise FileNotFoundError(f"GeoLife root not found: {geolife_root}")

    run_sensitivity_experiment(
        geolife_root=geolife_root,
        out_dir=out_dir,
        max_users=max_users,
        min_len=min_len,
        max_len=max_len,
        capability=capability,
        budget_grid=[2.0, 4.0, 6.0, 8.0, 10.0],
        risk_tau_grid=[0.4, 0.5, 0.6, 0.7, 0.8],
    )

    print("[INFO] Sensitivity experiment finished")


if __name__ == "__main__":
    main()
