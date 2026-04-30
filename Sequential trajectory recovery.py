# -*- coding: utf-8 -*-
"""
TwinkleGPS experiment code aligned with the paper:
- Static GI vs Twinkle vs Twinkle+Auditor for release-side utility
- Twinkle vs Twinkle+Auditor for GeoLife HMM sequential attack
- Baseline/burst schedule with public-intent-triggered burst windows
- Hard global budget filter
- Auditor chooses alpha_t and possibly suppresses releases
- GeoLife as primary benchmark; UrbanNav optional transfer case

"""

import os
import math
import random
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Global configuration
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Paths
# -------------------------
DATA_ROOT = r"E:\python\Geolife Trajectories 1.3\Geolife Trajectories 1.3\Data"
OUTPUT_DIR = r"E:\python\HMM-photo-twinkle4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional UrbanNav path.
# Expected CSV columns: time_s, lat, lon
URBANNAV_PATH = r""  # e.g. r"E:\python\UrbanNav\urban_nav_trace.csv"

# =========================================================
# Paper-aligned experimental protocol
# =========================================================

# -------------------------
# GeoLife data protocol
# Paper text:
# - retain trajectories with at least 80 samples
# - truncate each retained trace to at most 1200 samples
# - keep at most three trajectories per user
# -------------------------
MAX_USERS = 20
MAX_TRAJ_PER_USER = 3
MIN_TOTAL_POINTS = 80
MAX_POINTS_PER_TRAJECTORY = 1200

# We split each trajectory into train/test for HMM.
TRAIN_RATIO = 0.75
MIN_TRAIN_POINTS = 40
MIN_TEST_POINTS = 20

# Example plots
NUM_RANDOM_EXAMPLE_USERS = 3
EXAMPLE_MIN_TEST_POINTS = 10
EXAMPLE_MIN_HIGH_RISK_POINTS = 8
EXAMPLE_MAX_PLOT_POINTS = 120

# Cleaning
MIN_TIME_GAP_SEC = 0
MAX_SPEED_MPS = 65.0
MIN_MOVE_FOR_KEEP_M = 3.0

# Grid / HMM
GRID_SIZE_M = 250.0
MAX_CANDIDATE_STATES = 169
LOCAL_SEARCH_RADIUS_CELLS = 6
GLOBAL_FALLBACK_CANDIDATES = 40

# ---------------------------------------------------------
# Twinkle paper defaults from Section 4.1
# ---------------------------------------------------------
STATIC_INTERVAL_SEC = 10.0
STATIC_EPS = 0.06

BASELINE_INTERVAL_SEC = 30.0
BURST_INTERVAL_SEC = 2.0
BURST_LENGTH_SEC = 60.0
EPS_BASE = 0.03
EPS_BURST = 0.12
DEFAULT_BUDGET_B = 6.0

# Public synthetic intents at relative positions
INTENT_REL_POSITIONS = [0.25, 0.55, 0.80]

# GI / planar Laplace
SENSITIVITY_M = 200.0
MIN_EPS = 1e-6

# Auditor parameters
TAU = 0.60
TAU_HARD = 0.80
GAMMA = 10.0
ALPHA_MAX = 12.0

HIGH_RISK_THRESHOLD = TAU
MID_RISK_THRESHOLD = 0.45
# Simulated device-side estimate x_hat = x + bias + heavy-tailed noise
SIM_BIAS_STD_M = 6.0
SIM_LAPLACE_SCALE_M = 8.0
SIM_OUTLIER_PROB = 0.08
SIM_OUTLIER_SCALE_M = 20.0

# Place-risk proxy from sanitized dwell concentration
DWELL_RADIUS_M = 80.0

# Attack thresholds
HIGH_RISK_THRESHOLD = TAU
MID_RISK_THRESHOLD = 0.45

HIT_100M = 100.0
HIT_300M = 300.0
HIT_500M = 500.0

# Sensitivity sweeps from paper
SENS_BUDGETS = [2, 4, 6, 8, 10]
SENS_TAUS = [0.4, 0.5, 0.6, 0.7, 0.8]

# Plot style
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 10


# =========================================================
# Utility functions
# =========================================================

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def latlon_to_xy_m(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def xy_m_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110540.0
    lon = lon0 + x / (111320.0 * math.cos(math.radians(lat0)) + 1e-12)
    return lat, lon


def point_to_state(point, lat0, lon0, grid_size_m):
    lat, lon = point[:2]
    x, y = latlon_to_xy_m(lat, lon, lat0, lon0)
    gx = int(round(x / grid_size_m))
    gy = int(round(y / grid_size_m))
    return gx, gy


def state_to_center_latlon(state, lat0, lon0, grid_size_m):
    gx, gy = state
    x = gx * grid_size_m
    y = gy * grid_size_m
    return xy_m_to_latlon(x, y, lat0, lon0)


def trajectory_length_m(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        total += haversine_m(a[0], a[1], b[0], b[1])
    return total


def percentile_safe(arr, q):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, q))


def iqr_safe(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def median_safe(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.median(arr))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# GeoLife loading
# =========================================================

def parse_plt_file(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) <= 6:
            return []

        for line in lines[6:]:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                date_str = parts[5].strip()
                time_str = parts[6].strip()
                ts = pd.to_datetime(date_str + " " + time_str, errors="coerce")
                if pd.isna(ts):
                    continue
                rows.append((lat, lon, ts))
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")

    return rows


def preprocess_points(points):
    if not points:
        return []

    cleaned = [points[0]]
    for p in points[1:]:
        prev = cleaned[-1]
        dt = (p[2] - prev[2]).total_seconds()

        if dt < MIN_TIME_GAP_SEC:
            continue

        d = haversine_m(prev[0], prev[1], p[0], p[1])

        if d < MIN_MOVE_FOR_KEEP_M:
            continue

        if dt > 0:
            speed = d / dt
            if speed > MAX_SPEED_MPS:
                continue

        cleaned.append(p)

    return cleaned


def truncate_trajectory(points, max_points):
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in idx]


def load_user_trajectories(user_dir):
    traj_dir = os.path.join(user_dir, "Trajectory")
    if not os.path.isdir(traj_dir):
        return []

    files = [f for f in sorted(os.listdir(traj_dir)) if f.lower().endswith(".plt")]
    trajectories = []

    for fn in files:
        full_path = os.path.join(traj_dir, fn)
        pts = parse_plt_file(full_path)
        pts = preprocess_points(pts)

        if len(pts) < MIN_TOTAL_POINTS:
            continue

        pts = truncate_trajectory(pts, MAX_POINTS_PER_TRAJECTORY)
        trajectories.append(pts)

    return trajectories


def split_train_test_points(points, train_ratio):
    n = len(points)
    if n < MIN_TRAIN_POINTS + MIN_TEST_POINTS:
        return None, None

    k = int(n * train_ratio)
    k = max(MIN_TRAIN_POINTS, k)
    k = min(k, n - MIN_TEST_POINTS)

    if k <= 0 or k >= n:
        return None, None

    train_points = points[:k]
    test_points = points[k:]

    if len(train_points) < MIN_TRAIN_POINTS or len(test_points) < MIN_TEST_POINTS:
        return None, None

    return train_points, test_points


def load_geolife_subset(data_root, max_users=MAX_USERS, max_traj_per_user=MAX_TRAJ_PER_USER):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    user_ids = [u for u in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, u))]
    print(f"[INFO] Found {len(user_ids)} GeoLife user folders.")

    selected = []
    total_traces = 0

    for idx, uid in enumerate(user_ids, 1):
        print(f"\n[LOAD] User {uid} ({idx}/{len(user_ids)})")
        try:
            user_dir = os.path.join(data_root, uid)
            trajectories = load_user_trajectories(user_dir)

            valid_traj = []
            for tr in trajectories:
                train_points, test_points = split_train_test_points(tr, TRAIN_RATIO)
                if train_points is None:
                    continue
                valid_traj.append({
                    "all_points": tr,
                    "train_points": train_points,
                    "test_points": test_points,
                })

            if len(valid_traj) == 0:
                print("      skipped: no valid trajectories after split")
                continue

            valid_traj = valid_traj[:max_traj_per_user]
            selected.append((uid, valid_traj))
            total_traces += len(valid_traj)
            print(f"      accepted trajectories: {len(valid_traj)} | total so far: {total_traces}")

            if len(selected) >= max_users:
                print(f"[INFO] Reached MAX_USERS={max_users}.")
                break

        except Exception as e:
            print(f"[ERROR] User {uid} failed: {e}")
            traceback.print_exc()

    flat = []
    for uid, trajs in selected:
        for j, tr in enumerate(trajs, 1):
            flat.append((f"{uid}_traj{j}", uid, tr))

    print(f"\n[INFO] Loaded {len(selected)} users and {len(flat)} valid trajectories.")
    return flat


# =========================================================
# Optional UrbanNav loading
# =========================================================

def load_urbannav_trace(csv_path):
    """
    Expected CSV columns:
        time_s, lat, lon
    or:
        timestamp, lat, lon
    """
    if not csv_path or (not os.path.isfile(csv_path)):
        return None

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if "time_s" in cols:
        tcol = cols["time_s"]
    elif "timestamp" in cols:
        tcol = cols["timestamp"]
    else:
        raise ValueError("UrbanNav CSV must contain 'time_s' or 'timestamp'.")

    if "lat" not in cols or "lon" not in cols:
        raise ValueError("UrbanNav CSV must contain 'lat' and 'lon'.")

    lat_col = cols["lat"]
    lon_col = cols["lon"]

    rows = []
    for _, r in df.iterrows():
        rows.append((float(r[lat_col]), float(r[lon_col]), float(r[tcol])))

    rows = sorted(rows, key=lambda x: x[2])
    return rows


# =========================================================
# Simulated device-side estimate x_hat
# =========================================================

def simulate_internal_estimates(points, lat0, lon0, rng):
    """
    x_hat_t = x_t + static bias + heavy-tailed planar noise
    """
    bx = rng.normal(0.0, SIM_BIAS_STD_M)
    by = rng.normal(0.0, SIM_BIAS_STD_M)

    out = []
    for p in points:
        lat, lon, ts = p

        x, y = latlon_to_xy_m(lat, lon, lat0, lon0)

        nx = rng.laplace(0.0, SIM_LAPLACE_SCALE_M)
        ny = rng.laplace(0.0, SIM_LAPLACE_SCALE_M)

        if rng.random() < SIM_OUTLIER_PROB:
            nx += rng.laplace(0.0, SIM_OUTLIER_SCALE_M)
            ny += rng.laplace(0.0, SIM_OUTLIER_SCALE_M)

        xh = x + bx + nx
        yh = y + by + ny
        lat_h, lon_h = xy_m_to_latlon(xh, yh, lat0, lon0)
        out.append((lat_h, lon_h, ts))

    return out


# =========================================================
# Intent / burst windows
# =========================================================

def get_time_value(p):
    # GeoLife point uses pandas timestamp; UrbanNav may use float seconds
    ts = p[2]
    if isinstance(ts, pd.Timestamp):
        return ts.timestamp()
    return float(ts)


def build_synthetic_burst_windows(points, burst_length_sec=BURST_LENGTH_SEC):
    n = len(points)
    if n == 0:
        return []

    idxs = sorted(set(min(n - 1, max(0, int(round(r * (n - 1))))) for r in INTENT_REL_POSITIONS))
    start_times = [get_time_value(points[i]) for i in idxs]

    windows = []
    for st in start_times:
        windows.append((st, st + burst_length_sec))
    return windows


def is_in_burst_window(time_value, windows):
    for a, b in windows:
        if a <= time_value < b:
            return True
    return False


# =========================================================
# Training statistics for HMM / attacker
# =========================================================

def build_training_statistics(train_points_hat, lat0, lon0, grid_size_m):
    states = [point_to_state(p, lat0, lon0, grid_size_m) for p in train_points_hat]
    state_counts = Counter(states)

    transition_counts = defaultdict(Counter)
    for a, b in zip(states[:-1], states[1:]):
        transition_counts[a][b] += 1

    seen_states = set(states)
    return state_counts, transition_counts, seen_states


def build_user_hmm(train_points_hat, lat0, lon0, grid_size_m):
    states = [point_to_state(p, lat0, lon0, grid_size_m) for p in train_points_hat]
    state_counts = Counter(states)
    n_states = max(1, len(state_counts))

    start_counts = Counter([states[0]]) if states else Counter()

    transition_counts = defaultdict(Counter)
    for a, b in zip(states[:-1], states[1:]):
        transition_counts[a][b] += 1

    return {
        "states": list(state_counts.keys()),
        "state_counts": state_counts,
        "start_counts": start_counts,
        "transition_counts": transition_counts,
        "n_states": n_states,
        "lat0": lat0,
        "lon0": lon0,
        "grid_size_m": grid_size_m,
    }


# =========================================================
# Planar Laplace mechanism
# =========================================================

def add_planar_laplace_noise(point_hat, eps_t, lat0, lon0, sensitivity_m, rng):
    lat, lon = point_hat[:2]
    x, y = latlon_to_xy_m(lat, lon, lat0, lon0)

    b = sensitivity_m / max(eps_t, MIN_EPS)
    nx = rng.laplace(0.0, b)
    ny = rng.laplace(0.0, b)

    noisy_lat, noisy_lon = xy_m_to_latlon(x + nx, y + ny, lat0, lon0)
    return noisy_lat, noisy_lon


# =========================================================
# Auditor risk model: sanitized-history-only proxy
# =========================================================

def initialize_predictive_belief(hmm):
    counts = hmm["state_counts"]
    total = sum(counts.values())
    if total <= 0:
        states = hmm["states"]
        if not states:
            return {}
        p = 1.0 / len(states)
        return {s: p for s in states}
    return {s: c / total for s, c in counts.items()}


def predict_belief_one_step(q_prev, hmm):
    if not q_prev:
        return initialize_predictive_belief(hmm)

    q_bar = defaultdict(float)
    states = hmm["states"]
    n_states = max(1, hmm["n_states"])
    alpha = 0.5

    for s_prev, p_prev in q_prev.items():
        next_counts = hmm["transition_counts"].get(s_prev, {})
        total = sum(next_counts.values()) + alpha * n_states

        if total <= 0:
            continue

        # sparse transitions
        touched = set()
        for s_cur, cnt in next_counts.items():
            prob = (cnt + alpha) / total
            q_bar[s_cur] += p_prev * prob
            touched.add(s_cur)

        # small mass for unseen transitions
        residual_states = [s for s in states if s not in touched]
        if residual_states:
            prob_unseen = alpha / total
            for s_cur in residual_states:
                q_bar[s_cur] += p_prev * prob_unseen

    z = sum(q_bar.values())
    if z <= 0:
        return initialize_predictive_belief(hmm)
    return {s: v / z for s, v in q_bar.items()}


def posterior_update_from_release(q_bar, z_t, eps_eff, hmm, lat0, lon0):
    if z_t is None:
        return q_bar

    obs_x, obs_y = latlon_to_xy_m(z_t[0], z_t[1], lat0, lon0)

    scores = {}
    for s, prior_p in q_bar.items():
        st_lat, st_lon = state_to_center_latlon(s, lat0, lon0, GRID_SIZE_M)
        st_x, st_y = latlon_to_xy_m(st_lat, st_lon, lat0, lon0)
        d = math.hypot(obs_x - st_x, obs_y - st_y)

        # Plug-in heuristic from paper Eq. (26)
        ll = -max(eps_eff, MIN_EPS) * d
        scores[s] = math.log(max(prior_p, 1e-300)) + ll

    mx = max(scores.values()) if scores else 0.0
    vals = {s: math.exp(v - mx) for s, v in scores.items()}
    z = sum(vals.values())
    if z <= 0:
        return q_bar
    return {s: v / z for s, v in vals.items()}


def place_risk_from_sanitized_history(released_history, current_time_idx, lat0, lon0, lookback=20):
    """
    Local-window sanitized-history dwell concentration proxy.
    Still fully paper-compliant:
    - uses only sanitized history
    - causal
    - approximates sensitive-place / dwell concentration
    """
    hist = [z for z in released_history[-lookback:] if z is not None]
    if len(hist) < 3:
        return 0.0

    cur = hist[-1]
    cnt = 0
    total = 0

    for z in hist[:-1]:
        total += 1
        d = haversine_m(cur[0], cur[1], z[0], z[1])
        if d <= DWELL_RADIUS_M:
            cnt += 1

    if total <= 0:
        return 0.0

    return float(np.clip(cnt / total, 0.0, 1.0))


def compute_auditor_risk(q_bar, released_history, lat0, lon0):
    """
    Composite route/place risk.
    Paper says route-recoverability + sensitive-place exposure.
    We operationalize this as:
    - route risk = max predictive attacker state concentration
    - place risk = sanitized-history dwell concentration
    """
    route_risk = max(q_bar.values()) if q_bar else 0.0
    place_risk = place_risk_from_sanitized_history(released_history, len(released_history), lat0, lon0)

    # route + place combined
    R_t = 0.85 * route_risk + 0.15 * place_risk
    R_t = float(np.clip(R_t, 0.0, 1.0))
    return R_t, route_risk, place_risk


def auditor_alpha_from_risk(R_t, tau=TAU, gamma=GAMMA, alpha_max=ALPHA_MAX):
    alpha_t = min(alpha_max, max(1.0, 1.0 + gamma * max(0.0, R_t - tau)))
    return float(alpha_t)


# =========================================================
# HMM attacker
# =========================================================

def log_startprob(state, hmm):
    alpha = 1.0
    total = sum(hmm["start_counts"].values()) + alpha * hmm["n_states"]
    num = hmm["start_counts"].get(state, 0) + alpha
    return math.log(num / total)


def log_transprob(s_prev, s_cur, hmm):
    alpha = 0.5
    next_counts = hmm["transition_counts"].get(s_prev, {})
    total = sum(next_counts.values()) + alpha * hmm["n_states"]
    num = next_counts.get(s_cur, 0) + alpha
    return math.log(num / total)


def emission_logprob(observed_point, hidden_state, lat0, lon0, eps_t, sensitivity_m, sigma_floor=100.0):
    obs_x, obs_y = latlon_to_xy_m(observed_point[0], observed_point[1], lat0, lon0)
    st_lat, st_lon = state_to_center_latlon(hidden_state, lat0, lon0, GRID_SIZE_M)
    st_x, st_y = latlon_to_xy_m(st_lat, st_lon, lat0, lon0)

    d = math.hypot(obs_x - st_x, obs_y - st_y)
    sigma_m = max(sigma_floor, 1.8 * sensitivity_m / max(eps_t, MIN_EPS))
    return -0.5 * (d / sigma_m) ** 2


def candidate_states_for_observation(observed_point, hmm):
    states = hmm["states"]
    if len(states) <= MAX_CANDIDATE_STATES:
        return states

    lat0 = hmm["lat0"]
    lon0 = hmm["lon0"]
    gx, gy = point_to_state(observed_point, lat0, lon0, hmm["grid_size_m"])

    state_counts = hmm["state_counts"]
    local_candidates = []

    for s in states:
        if abs(s[0] - gx) <= LOCAL_SEARCH_RADIUS_CELLS and abs(s[1] - gy) <= LOCAL_SEARCH_RADIUS_CELLS:
            local_candidates.append(s)

    local_candidates = sorted(local_candidates, key=lambda s: -state_counts[s])
    global_candidates = [s for s, _ in state_counts.most_common(GLOBAL_FALLBACK_CANDIDATES)]

    merged = []
    seen = set()

    for s in local_candidates + global_candidates:
        if s not in seen:
            merged.append(s)
            seen.add(s)
        if len(merged) >= MAX_CANDIDATE_STATES:
            break

    if not merged:
        merged = [s for s, _ in state_counts.most_common(MAX_CANDIDATE_STATES)]

    return merged[:MAX_CANDIDATE_STATES]


def viterbi_decode(observed_points, eps_series, hmm):
    if len(observed_points) == 0:
        return []

    lat0 = hmm["lat0"]
    lon0 = hmm["lon0"]

    candidates_list = [candidate_states_for_observation(obs, hmm) for obs in observed_points]

    dp = []
    back = []

    cand0 = candidates_list[0]
    dp0 = {}
    back0 = {}
    for s in cand0:
        dp0[s] = log_startprob(s, hmm) + emission_logprob(
            observed_points[0], s, lat0, lon0, eps_series[0], SENSITIVITY_M
        )
        back0[s] = None
    dp.append(dp0)
    back.append(back0)

    for t in range(1, len(observed_points)):
        cand_prev = candidates_list[t - 1]
        cand_cur = candidates_list[t]
        dpt = {}
        backt = {}

        for s_cur in cand_cur:
            emit = emission_logprob(observed_points[t], s_cur, lat0, lon0, eps_series[t], SENSITIVITY_M)
            best_score = -1e100
            best_prev = None

            for s_prev in cand_prev:
                prev_score = dp[t - 1].get(s_prev, -1e100)
                score = prev_score + log_transprob(s_prev, s_cur, hmm) + emit
                if score > best_score:
                    best_score = score
                    best_prev = s_prev

            dpt[s_cur] = best_score
            backt[s_cur] = best_prev

        dp.append(dpt)
        back.append(backt)

    last_states = dp[-1]
    if not last_states:
        return []

    sT = max(last_states, key=last_states.get)
    path = [sT]

    for t in range(len(observed_points) - 1, 0, -1):
        sT = back[t][sT]
        if sT is None:
            break
        path.append(sT)

    path.reverse()

    while len(path) < len(observed_points):
        path.append(path[-1])

    return path


def states_to_points(states, lat0, lon0, grid_size_m):
    return [state_to_center_latlon(s, lat0, lon0, grid_size_m) for s in states]


# =========================================================
# Schedules
# =========================================================

def static_should_emit(last_emit_time, current_time):
    if last_emit_time is None:
        return True
    return (current_time - last_emit_time) >= STATIC_INTERVAL_SEC


def twinkle_mode_and_nominal_eps(current_time, burst_windows):
    in_burst = is_in_burst_window(current_time, burst_windows)
    if in_burst:
        return 1, EPS_BURST, BURST_INTERVAL_SEC
    return 0, EPS_BASE, BASELINE_INTERVAL_SEC


def twinkle_should_emit(last_emit_time, current_time, interval_sec):
    if last_emit_time is None:
        return True
    return (current_time - last_emit_time) >= interval_sec


# =========================================================
# Release pipelines
# =========================================================

def run_static_gi_release(trace_id, true_points, hat_points, lat0, lon0, rng, budget_B=DEFAULT_BUDGET_B, burst_windows=None):
    records = []
    released_points = []
    eps_series = []
    emit_truth_points = []

    cum_budget = 0.0
    last_emit_time = None

    for t, (true_p, hat_p) in enumerate(zip(true_points, hat_points)):
        current_time = get_time_value(true_p)
        in_burst = is_in_burst_window(current_time, burst_windows or [])

        candidate = static_should_emit(last_emit_time, current_time)
        a_t = 0
        y_t = None
        eps_eff = np.nan

        if candidate:
            if cum_budget + STATIC_EPS <= budget_B:
                a_t = 1
                eps_eff = STATIC_EPS
                z_lat, z_lon = add_planar_laplace_noise(hat_p, eps_eff, lat0, lon0, SENSITIVITY_M, rng)
                y_t = (z_lat, z_lon)
                released_points.append(y_t)
                eps_series.append(eps_eff)
                emit_truth_points.append(true_p)
                cum_budget += eps_eff
                last_emit_time = current_time

        rel_err = haversine_m(true_p[0], true_p[1], y_t[0], y_t[1]) if y_t is not None else np.nan

        records.append({
            "trace_id": trace_id,
            "t": t,
            "time": current_time,
            "method": "Static GI",
            "mode": 1 if in_burst else 0,  # only for reference burst membership
            "candidate": int(candidate),
            "emit": int(a_t),
            "suppressed": int(candidate and not a_t),
            "true_lat": true_p[0],
            "true_lon": true_p[1],
            "hat_lat": hat_p[0],
            "hat_lon": hat_p[1],
            "z_lat": y_t[0] if y_t is not None else np.nan,
            "z_lon": y_t[1] if y_t is not None else np.nan,
            "eps_nom": STATIC_EPS if candidate else np.nan,
            "alpha_t": 1.0 if candidate else np.nan,
            "eps_eff": eps_eff,
            "cum_budget": cum_budget,
            "release_error": rel_err,
            "in_burst_window": int(in_burst),
            "R_t": np.nan,
            "route_risk": np.nan,
            "place_risk": np.nan,
        })

    return pd.DataFrame(records), released_points, eps_series, emit_truth_points


def run_twinkle_release(
    trace_id,
    true_points,
    hat_points,
    lat0,
    lon0,
    rng,
    hmm,
    burst_windows,
    budget_B=DEFAULT_BUDGET_B,
    with_auditor=False,
    tau=TAU,
    tau_hard=TAU_HARD,
    gamma=GAMMA,
    alpha_max=ALPHA_MAX,
):
    records = []
    released_points = []
    eps_series = []
    emit_truth_points = []

    cum_budget = 0.0
    last_emit_time = None

    # sanitized-history-only auditor state
    released_history = []
    q_prev = initialize_predictive_belief(hmm)

    for t, (true_p, hat_p) in enumerate(zip(true_points, hat_points)):
        current_time = get_time_value(true_p)

        mode_t, eps_nom, interval_sec = twinkle_mode_and_nominal_eps(current_time, burst_windows)
        candidate = twinkle_should_emit(last_emit_time, current_time, interval_sec)

        q_bar = predict_belief_one_step(q_prev, hmm)
        R_t, route_risk, place_risk = compute_auditor_risk(q_bar, released_history, lat0, lon0)

        alpha_t = 1.0
        a_t = 0
        y_t = None
        eps_eff = np.nan
        suppressed = 0

        if candidate:
            if with_auditor:
                alpha_t = auditor_alpha_from_risk(R_t, tau=tau, gamma=gamma, alpha_max=alpha_max)
                eps_eff_try = max(MIN_EPS, eps_nom / alpha_t)

                # hard risk shutdown
                if R_t > tau_hard:
                    a_t = 0
                    suppressed = 1
                else:
                    # budget filter: first try current alpha;
                    # if budget fails, increase alpha up to alpha_max;
                    # if still fails, suppress.
                    if cum_budget + eps_eff_try <= budget_B:
                        a_t = 1
                        eps_eff = eps_eff_try
                    else:
                        # additional inflation to fit remaining budget
                        remain = budget_B - cum_budget
                        if remain > 0:
                            alpha_need = eps_nom / max(remain, MIN_EPS)
                            alpha_t = min(alpha_max, max(alpha_t, alpha_need))
                            eps_eff_try2 = max(MIN_EPS, eps_nom / alpha_t)
                            if cum_budget + eps_eff_try2 <= budget_B:
                                a_t = 1
                                eps_eff = eps_eff_try2
                            else:
                                a_t = 0
                                suppressed = 1
                        else:
                            a_t = 0
                            suppressed = 1
            else:
                alpha_t = 1.0
                eps_eff_try = eps_nom
                if cum_budget + eps_eff_try <= budget_B:
                    a_t = 1
                    eps_eff = eps_eff_try
                else:
                    a_t = 0
                    suppressed = 1

            if a_t == 1:
                z_lat, z_lon = add_planar_laplace_noise(hat_p, eps_eff, lat0, lon0, SENSITIVITY_M, rng)
                y_t = (z_lat, z_lon)
                released_points.append(y_t)
                eps_series.append(eps_eff)
                emit_truth_points.append(true_p)
                cum_budget += eps_eff
                last_emit_time = current_time

        # update posterior only from sanitized release / suppression
        q_prev = posterior_update_from_release(q_bar, y_t, eps_eff if np.isfinite(eps_eff) else eps_nom, hmm, lat0, lon0)
        released_history.append(y_t)

        rel_err = haversine_m(true_p[0], true_p[1], y_t[0], y_t[1]) if y_t is not None else np.nan

        records.append({
            "trace_id": trace_id,
            "t": t,
            "time": current_time,
            "method": "Twinkle+Auditor" if with_auditor else "Twinkle",
            "mode": mode_t,
            "candidate": int(candidate),
            "emit": int(a_t),
            "suppressed": int(suppressed),
            "true_lat": true_p[0],
            "true_lon": true_p[1],
            "hat_lat": hat_p[0],
            "hat_lon": hat_p[1],
            "z_lat": y_t[0] if y_t is not None else np.nan,
            "z_lon": y_t[1] if y_t is not None else np.nan,
            "eps_nom": eps_nom if candidate else np.nan,
            "alpha_t": alpha_t if candidate else np.nan,
            "eps_eff": eps_eff,
            "cum_budget": cum_budget,
            "release_error": rel_err,
            "in_burst_window": int(mode_t == 1),
            "R_t": R_t,
            "route_risk": route_risk,
            "place_risk": place_risk,
            "high_risk": int(R_t > HIGH_RISK_THRESHOLD),
            "mid_or_high_risk": int(R_t >= MID_RISK_THRESHOLD),
            "risk_level": "high" if R_t > HIGH_RISK_THRESHOLD else ("mid" if R_t >= MID_RISK_THRESHOLD else "low"),
        })

    return pd.DataFrame(records), released_points, eps_series, emit_truth_points


# =========================================================
# Attack evaluation
# =========================================================

def evaluate_attack(trace_id, variant_name, emitted_true_hat_points, released_points, eps_series, hmm, lat0, lon0, release_df):
    """
    emitted_true_hat_points: list of tuples (true_point, hat_point, original_t)
    Only emitted points are attack-evaluated, consistent with paper wording.
    """
    if len(released_points) == 0:
        return pd.DataFrame(), []

    true_hat_points = emitted_true_hat_points
    true_hat_only = [x[1] for x in true_hat_points]  # x_hat_t for exact-state comparison
    true_points = [x[0] for x in true_hat_points]
    orig_t_list = [x[2] for x in true_hat_points]

    pred_states = viterbi_decode(released_points, eps_series, hmm)
    pred_points = states_to_points(pred_states, lat0, lon0, GRID_SIZE_M)

    n = min(len(true_points), len(pred_points), len(pred_states))
    rows = []

    release_lookup = release_df.set_index("t")

    for i in range(n):
        true_p = true_points[i]
        hat_p = true_hat_only[i]
        pred_p = pred_points[i]
        pred_s = pred_states[i]
        orig_t = orig_t_list[i]

        true_state = point_to_state(hat_p, lat0, lon0, GRID_SIZE_M)
        err = haversine_m(true_p[0], true_p[1], pred_p[0], pred_p[1])

        rinfo = release_lookup.loc[orig_t]

        rows.append({
            "trace_id": trace_id,
            "variant": variant_name,
            "emit_idx": i,
            "t": orig_t,
            "true_lat": true_p[0],
            "true_lon": true_p[1],
            "hat_lat": hat_p[0],
            "hat_lon": hat_p[1],
            "pred_lat": pred_p[0],
            "pred_lon": pred_p[1],
            "exact": int(pred_s == true_state),
            "hit_100m": int(err <= HIT_100M),
            "hit_300m": int(err <= HIT_300M),
            "hit_500m": int(err <= HIT_500M),
            "recover_error": err,
            "R_t": rinfo.get("R_t", np.nan),
            "route_risk": rinfo.get("route_risk", np.nan),
            "place_risk": rinfo.get("place_risk", np.nan),
            "high_risk": rinfo.get("high_risk", 0),
            "mid_or_high_risk": rinfo.get("mid_or_high_risk", 0),
            "risk_level": rinfo.get("risk_level", "na"),
            "release_error": rinfo.get("release_error", np.nan),
            "mode": rinfo.get("mode", 0),
            "in_burst_window": rinfo.get("in_burst_window", 0),
        })

    attack_df = pd.DataFrame(rows)
    return attack_df, pred_points


# =========================================================
# Utility summaries
# =========================================================

def summarize_release_errors_for_method(df_method, scope):
    if scope == "All":
        sub = df_method[df_method["emit"] == 1]
    elif scope == "Burst":
        sub = df_method[(df_method["emit"] == 1) & (df_method["in_burst_window"] == 1)]
    else:
        raise ValueError("scope must be All or Burst")

    errs = sub["release_error"].dropna().values
    return {
        "n": int(len(sub)),
        "median": median_safe(errs),
        "p95": percentile_safe(errs, 95),
    }


def make_geolife_release_summary(release_all_df):
    rows = []
    for scope in ["All", "Burst"]:
        for method in ["Static GI", "Twinkle", "Twinkle+Auditor"]:
            subm = release_all_df[release_all_df["method"] == method]
            s = summarize_release_errors_for_method(subm, scope)
            rows.append({
                "Scope": scope,
                "Method": method,
                "n": s["n"],
                "Median": s["median"],
                "p95": s["p95"],
            })
    return pd.DataFrame(rows)


def make_attack_summary_table(attack_df, release_df_twinkle, release_df_aud):
    rows = []

    def add_block(subset_name, df_no, df_wi):
        rows.extend([
            {"Subset": subset_name, "Metric": "Exact (%)",
             "Twinkle": 100 * df_no["exact"].mean(), "Twinkle+Auditor": 100 * df_wi["exact"].mean()},
            {"Subset": subset_name, "Metric": "Hit@100 m (%)",
             "Twinkle": 100 * df_no["hit_100m"].mean(), "Twinkle+Auditor": 100 * df_wi["hit_100m"].mean()},
            {"Subset": subset_name, "Metric": "Hit@300 m (%)",
             "Twinkle": 100 * df_no["hit_300m"].mean(), "Twinkle+Auditor": 100 * df_wi["hit_300m"].mean()},
            {"Subset": subset_name, "Metric": "Hit@500 m (%)",
             "Twinkle": 100 * df_no["hit_500m"].mean(), "Twinkle+Auditor": 100 * df_wi["hit_500m"].mean()},
            {"Subset": subset_name, "Metric": r"Median e_rec (m)",
             "Twinkle": median_safe(df_no["recover_error"]), "Twinkle+Auditor": median_safe(df_wi["recover_error"])},
            {"Subset": subset_name, "Metric": r"IQR e_rec (m)",
             "Twinkle": iqr_safe(df_no["recover_error"]), "Twinkle+Auditor": iqr_safe(df_wi["recover_error"])},
        ])

    df_no = attack_df[attack_df["variant"] == "Twinkle"]
    df_wi = attack_df[attack_df["variant"] == "Twinkle+Auditor"]

    add_block("All", df_no, df_wi)

    hs_no = df_no[df_no["high_risk"] == 1]
    hs_wi = df_wi[df_wi["high_risk"] == 1]
    add_block("High-risk", hs_no, hs_wi)

    rel_no = release_df_twinkle[release_df_twinkle["emit"] == 1]["release_error"].dropna().values
    rel_wi = release_df_aud[release_df_aud["emit"] == 1]["release_error"].dropna().values
    rows.append({
        "Subset": "Release",
        "Metric": "Median e_rel (m)",
        "Twinkle": median_safe(rel_no),
        "Twinkle+Auditor": median_safe(rel_wi),
    })

    out = pd.DataFrame(rows)
    out["Change"] = out["Twinkle+Auditor"] - out["Twinkle"]
    return out


# =========================================================
# Sensitivity analysis
# =========================================================

def run_single_trace_for_sensitivity(
    true_points, hat_points, lat0, lon0, burst_windows, hmm, rng_seed, budget_B=DEFAULT_BUDGET_B, tau=TAU
):
    rng = np.random.default_rng(rng_seed)
    release_df, _, _, _ = run_twinkle_release(
        trace_id="sens",
        true_points=true_points,
        hat_points=hat_points,
        lat0=lat0,
        lon0=lon0,
        rng=rng,
        hmm=hmm,
        burst_windows=burst_windows,
        budget_B=budget_B,
        with_auditor=True,
        tau=tau,
        tau_hard=TAU_HARD,
        gamma=GAMMA,
        alpha_max=ALPHA_MAX,
    )

    emit_df = release_df[release_df["emit"] == 1]
    total_span = max(get_time_value(true_points[-1]) - get_time_value(true_points[0]), 1.0)
    emit_rate = len(emit_df) / total_span
    budget_used = emit_df["eps_eff"].sum()
    med = median_safe(emit_df["release_error"])
    p95 = percentile_safe(emit_df["release_error"], 95)
    mean_R = emit_df["R_t"].mean() if len(emit_df) else np.nan

    return {
        "emit_rate": emit_rate,
        "budget_used": budget_used,
        "median_err": med,
        "p95_err": p95,
        "mean_R_t": mean_R,
    }


# =========================================================
# Plotting: utility CDFs
# =========================================================

def plot_cdf(ax, arr, label):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    xs = np.sort(arr)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, label=label, linewidth=2)


def label_with_stats(name, arr):
    med = median_safe(arr)
    p95 = percentile_safe(arr, 95)
    return f"{name} (med={med:.2f}m, p95={p95:.2f}m)"


def plot_geolife_utility_cdfs(release_df, output_dir):
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["Static GI", "Twinkle", "Twinkle+Auditor"]:
        errs = release_df[(release_df["method"] == method) & (release_df["emit"] == 1)]["release_error"].dropna().values
        plot_cdf(ax, errs, label_with_stats(method, errs))
    ax.set_xlabel("Position error ||z_t - x_t|| (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Experiment 3 (GeoLife) — Position error CDF at all emission times")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_cdf_l1_all_geolife.png"), dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["Static GI", "Twinkle", "Twinkle+Auditor"]:
        errs = release_df[
            (release_df["method"] == method) &
            (release_df["emit"] == 1) &
            (release_df["in_burst_window"] == 1)
        ]["release_error"].dropna().values
        plot_cdf(ax, errs, label_with_stats(method, errs))
    ax.set_xlabel("Position error ||z_t - x_t|| (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Experiment 3 (GeoLife) — Position error CDF restricted to burst windows")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_cdf_l1_burst_geolife.png"), dpi=220)
    plt.close()


def plot_urbannav_utility_cdfs(release_df, output_dir):
    if release_df is None or len(release_df) == 0:
        print("[PLOT] UrbanNav utility CDF skipped: no UrbanNav data.")
        return

    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["Static GI", "Twinkle", "Twinkle+Auditor"]:
        errs = release_df[(release_df["method"] == method) & (release_df["emit"] == 1)]["release_error"].dropna().values
        plot_cdf(ax, errs, label_with_stats(method, errs))
    ax.set_xlabel("Position error ||z_t - x_t|| (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Experiment 1a — Position error CDF at emission times")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_cdf_l1_all.png"), dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ["Static GI", "Twinkle", "Twinkle+Auditor"]:
        errs = release_df[
            (release_df["method"] == method) &
            (release_df["emit"] == 1) &
            (release_df["in_burst_window"] == 1)
        ]["release_error"].dropna().values
        plot_cdf(ax, errs, label_with_stats(method, errs))
    ax.set_xlabel("Position error ||z_t - x_t|| (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Experiment 1b — CDF restricted to Twinkle burst windows")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_cdf_l1_burst.png"), dpi=220)
    plt.close()


# =========================================================
# Plotting: attack figures (paper Fig. 2 / 3)
# =========================================================

def plot_overall_attack_bars(attack_df, output_dir):
    metrics = ["exact", "hit_100m", "hit_300m", "hit_500m"]
    labels = ["Exact", "Hit@100m", "Hit@300m", "Hit@500m"]

    vals_no = []
    vals_with = []

    for m in metrics:
        vals_no.append(attack_df[attack_df["variant"] == "Twinkle"][m].mean())
        vals_with.append(attack_df[attack_df["variant"] == "Twinkle+Auditor"][m].mean())

    x = np.arange(len(metrics))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vals_no, width, label="Twinkle")
    plt.bar(x + width / 2, vals_with, width, label="Twinkle+Auditor")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Attack success rate")
    plt.title("Overall attack success comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_overall_attack_success.png"), dpi=220)
    plt.close()


def plot_highrisk_attack_bars(attack_df, output_dir):
    high_df = attack_df[attack_df["high_risk"] == 1]

    metrics = ["exact", "hit_100m", "hit_300m", "hit_500m"]
    labels = ["Exact", "Hit@100m", "Hit@300m", "Hit@500m"]

    vals_no = []
    vals_with = []

    for m in metrics:
        vals_no.append(high_df[high_df["variant"] == "Twinkle"][m].mean() if len(high_df) else np.nan)
        vals_with.append(high_df[high_df["variant"] == "Twinkle+Auditor"][m].mean() if len(high_df) else np.nan)

    x = np.arange(len(metrics))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, vals_no, width, label="Twinkle")
    plt.bar(x + width / 2, vals_with, width, label="Twinkle+Auditor")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Attack success rate")
    plt.title("High-risk attack success comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_highrisk_attack_success.png"), dpi=220)
    plt.close()


def plot_recovery_error_boxplots(attack_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    overall_no = attack_df[attack_df["variant"] == "Twinkle"]["recover_error"].dropna().values
    overall_with = attack_df[attack_df["variant"] == "Twinkle+Auditor"]["recover_error"].dropna().values

    axes[0].boxplot([overall_no, overall_with], tick_labels=["Twinkle", "Twinkle+Auditor"], showfliers=False)
    axes[0].set_title("Overall attacker reconstruction error")
    axes[0].set_ylabel("Meters")

    high_df = attack_df[attack_df["high_risk"] == 1]
    high_no = high_df[high_df["variant"] == "Twinkle"]["recover_error"].dropna().values
    high_with = high_df[high_df["variant"] == "Twinkle+Auditor"]["recover_error"].dropna().values

    if len(high_no) > 0 and len(high_with) > 0:
        axes[1].boxplot([high_no, high_with], tick_labels=["Twinkle", "Twinkle+Auditor"], showfliers=False)
    else:
        axes[1].text(0.5, 0.5, "No high-risk samples", ha="center", va="center")
    axes[1].set_title("High-risk attacker reconstruction error")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_recovery_error_boxplots.png"), dpi=220)
    plt.close()


def plot_release_error_boxplots(release_twinkle_df, release_twinkle_aud_df, output_dir):
    plt.figure(figsize=(8, 5))
    data = [
        release_twinkle_df[release_twinkle_df["emit"] == 1]["release_error"].dropna().values,
        release_twinkle_aud_df[release_twinkle_aud_df["emit"] == 1]["release_error"].dropna().values,
    ]
    plt.boxplot(data, tick_labels=["Twinkle", "Twinkle+Auditor"], showfliers=False)
    plt.ylabel("Release distortion (meters)")
    plt.title("Release error comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_release_error_boxplot.png"), dpi=220)
    plt.close()


# =========================================================
# Plotting: sensitivity (paper Fig. 4)
# =========================================================

def plot_sensitivity_budget(df_budget, output_dir):
    plt.figure(figsize=(8, 6))
    x = df_budget["B"].values

    plt.plot(x, df_budget["median_err"], marker="o", label="Median err (left)")
    plt.plot(x, df_budget["p95_err"], marker="o", label="p95 err (left)")
    plt.plot(x, df_budget["emit_rate"] * 1000, marker="o", label="Emit rate ×1000")
    plt.plot(x, df_budget["budget_used"], marker="o", label="Budget used")

    plt.xlabel("Budget (B)")
    plt.ylabel("Value")
    plt.title("Utility sensitivity overview — budget B (L1-only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_sensitivity_budget.png"), dpi=220)
    plt.close()


def plot_sensitivity_tau(df_tau, output_dir):
    plt.figure(figsize=(8, 6))
    x = df_tau["tau"].values

    plt.plot(x, df_tau["median_err"], marker="o", label="Median err")
    plt.plot(x, df_tau["p95_err"], marker="o", label="p95 err")
    plt.plot(x, df_tau["emit_rate"] * 1000, marker="o", label="Emit rate ×1000")
    plt.plot(x, df_tau["mean_R_t"] * 100, marker="o", label="Mean R_t ×100")

    plt.xlabel("Auditor threshold tau")
    plt.ylabel("Value")
    plt.title("Utility sensitivity overview — tau (L1-only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_sensitivity_tau.png"), dpi=220)
    plt.close()


# =========================================================
# Plotting: UrbanNav case study (paper Fig. 5)
# =========================================================

def plot_urbannav_case_study(release_df, output_dir):
    if release_df is None or len(release_df) == 0:
        print("[PLOT] UrbanNav case study skipped: no UrbanNav data.")
        return

    df = release_df[release_df["method"] == "Twinkle+Auditor"].copy()
    if len(df) == 0:
        return

    t = df["time"].values
    eps_eff = df["eps_eff"].fillna(0).values
    cum_budget = df["cum_budget"].values
    mode = df["mode"].values
    emit = df["emit"].values

    plt.figure(figsize=(8, 4))
    plt.plot(t, cum_budget)
    plt.xlabel("time (s)")
    plt.ylabel("budget")
    plt.title("Experiment 2 — Twinkle with auditor — budget used E_t")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_urbannav_case_budget.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(t, eps_eff)
    plt.xlabel("time (s)")
    plt.ylabel("eps_eff")
    plt.title("Experiment 2 — Twinkle with auditor — eps_eff")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_urbannav_case_eps.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, mode, label="mode (0=base,1=burst)")
    plt.plot(t, emit, label="a_t (emit)")
    plt.xlabel("time (s)")
    plt.ylabel("indicator")
    plt.title("Experiment 2 — Twinkle with auditor — mode & a_t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_urbannav_case_mode_release.png"), dpi=220)
    plt.close()


# =========================================================
# Plotting: example trajectories
# =========================================================

def plot_example_trajectory(trace_id, true_points, release_tw_df, release_aud_df, attack_tw_df, attack_aud_df, output_dir, max_plot_points=120):
    n = min(max_plot_points, len(true_points))
    if n <= 5:
        return

    true_sub = true_points[:n]
    true_lats = [p[0] for p in true_sub]
    true_lons = [p[1] for p in true_sub]

    rel_tw_sub = release_tw_df.iloc[:n].copy()
    rel_aud_sub = release_aud_df.iloc[:n].copy()

    tw_emit = rel_tw_sub[rel_tw_sub["emit"] == 1]
    aud_emit = rel_aud_sub[rel_aud_sub["emit"] == 1]

    plt.figure(figsize=(10, 8))
    plt.plot(true_lons, true_lats, "-k", linewidth=2.2, label="True trajectory", zorder=5)
    plt.scatter(tw_emit["z_lon"], tw_emit["z_lat"], s=14, alpha=0.55, color="tab:blue", label="Twinkle releases", zorder=3)
    plt.scatter(aud_emit["z_lon"], aud_emit["z_lat"], s=14, alpha=0.55, color="tab:orange", label="Twinkle+Auditor releases", zorder=3)

    if len(attack_tw_df) > 0:
        atk = attack_tw_df.copy()
        plt.plot(atk["pred_lon"], atk["pred_lat"], "--", color="red", linewidth=1.6, label="Attacker reconstruction Twinkle", zorder=4)

    if len(attack_aud_df) > 0:
        atk = attack_aud_df.copy()
        plt.plot(atk["pred_lon"], atk["pred_lat"], "--", color="blue", linewidth=1.6, label="Attacker reconstruction Twinkle+Auditor", zorder=4)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Example trajectory comparison for {trace_id}")
    plt.legend(loc="lower left", framealpha=0.9)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig10_example_trajectory_{trace_id}.png")
    plt.savefig(save_path, dpi=220)
    plt.close()


# =========================================================
# Main experiment pipeline
# =========================================================

def run_geolife_main_experiment():
    flat_traces = load_geolife_subset(DATA_ROOT, MAX_USERS, MAX_TRAJ_PER_USER)
    if len(flat_traces) == 0:
        raise RuntimeError("No valid GeoLife trajectories found.")

    all_release = []
    all_attack = []
    example_candidates = []

    for idx, (trace_id, uid, tr) in enumerate(flat_traces, 1):
        print(f"\n[RUN] GeoLife trace {trace_id} ({idx}/{len(flat_traces)})")

        all_points = tr["all_points"]
        train_points = tr["train_points"]
        test_points = tr["test_points"]

        lat0 = all_points[0][0]
        lon0 = all_points[0][1]

        rng = np.random.default_rng(SEED + idx)

        train_hat = simulate_internal_estimates(train_points, lat0, lon0, rng)
        test_hat = simulate_internal_estimates(test_points, lat0, lon0, rng)

        hmm = build_user_hmm(train_hat, lat0, lon0, GRID_SIZE_M)
        burst_windows = build_synthetic_burst_windows(test_points, BURST_LENGTH_SEC)

        # Static GI
        static_df, static_rels, static_eps, static_true_emit = run_static_gi_release(
            trace_id, test_points, test_hat, lat0, lon0, rng, DEFAULT_BUDGET_B, burst_windows
        )

        # Twinkle
        tw_df, tw_rels, tw_eps, tw_true_emit = run_twinkle_release(
            trace_id, test_points, test_hat, lat0, lon0, rng, hmm, burst_windows,
            budget_B=DEFAULT_BUDGET_B, with_auditor=False
        )

        # Twinkle + Auditor
        aud_df, aud_rels, aud_eps, aud_true_emit = run_twinkle_release(
            trace_id, test_points, test_hat, lat0, lon0, rng, hmm, burst_windows,
            budget_B=DEFAULT_BUDGET_B, with_auditor=True
        )

        # Attach emitted point mappings for attack
        tw_emit_idx = tw_df[tw_df["emit"] == 1]["t"].tolist()
        aud_emit_idx = aud_df[aud_df["emit"] == 1]["t"].tolist()

        tw_emitted_true_hat_points = [(test_points[t], test_hat[t], t) for t in tw_emit_idx]
        aud_emitted_true_hat_points = [(test_points[t], test_hat[t], t) for t in aud_emit_idx]

        atk_tw, _ = evaluate_attack(trace_id, "Twinkle", tw_emitted_true_hat_points, tw_rels, tw_eps, hmm, lat0, lon0, tw_df)
        atk_aud, _ = evaluate_attack(trace_id, "Twinkle+Auditor", aud_emitted_true_hat_points, aud_rels, aud_eps, hmm, lat0, lon0, aud_df)

        rel_trace_df = pd.concat([static_df, tw_df, aud_df], ignore_index=True)
        atk_trace_df = pd.concat([atk_tw, atk_aud], ignore_index=True)

        all_release.append(rel_trace_df)
        all_attack.append(atk_trace_df)

        if len(test_points) >= EXAMPLE_MIN_TEST_POINTS and aud_df["high_risk"].sum() >= EXAMPLE_MIN_HIGH_RISK_POINTS:
            example_candidates.append((trace_id, test_points, tw_df.copy(), aud_df.copy(), atk_tw.copy(), atk_aud.copy()))

        print(
            f"      Static emits={static_df['emit'].sum()} | "
            f"Twinkle emits={tw_df['emit'].sum()} | "
            f"Twinkle+Aud emits={aud_df['emit'].sum()} | "
            f"High-risk in aud={aud_df['high_risk'].sum()}"
        )

    release_all_df = pd.concat(all_release, ignore_index=True)
    attack_all_df = pd.concat(all_attack, ignore_index=True)

    # Main utility table for GeoLife
    geolife_release_summary = make_geolife_release_summary(release_all_df)

    # Main attack table
    release_tw = release_all_df[release_all_df["method"] == "Twinkle"].copy()
    release_aud = release_all_df[release_all_df["method"] == "Twinkle+Auditor"].copy()
    attack_summary = make_attack_summary_table(attack_all_df, release_tw, release_aud)

    # Example payloads
    example_payloads = []
    if len(example_candidates) > 0:
        rng_examples = random.Random(SEED + 999)
        example_candidates_sorted = sorted(
            example_candidates,
            key=lambda x: (len(x[1]), x[3]["high_risk"].sum()),
            reverse=True
        )
        main_example = example_candidates_sorted[0]
        remaining = example_candidates_sorted[1:]
        num_extra = min(NUM_RANDOM_EXAMPLE_USERS, len(remaining))
        random_examples = rng_examples.sample(remaining, num_extra) if num_extra > 0 else []
        example_payloads = [main_example] + random_examples

    return release_all_df, attack_all_df, geolife_release_summary, attack_summary, example_payloads, flat_traces


def run_geolife_sensitivity(flat_traces):
    budget_rows = []
    tau_rows = []

    # Use up to first 75 traces if available, per paper wording
    sens_traces = flat_traces[:75]

    # Panel A: vary B, fixed tau=0.6
    for B in SENS_BUDGETS:
        vals = []
        for idx, (trace_id, uid, tr) in enumerate(sens_traces, 1):
            all_points = tr["all_points"]
            test_points = tr["test_points"]
            train_points = tr["train_points"]
            lat0 = all_points[0][0]
            lon0 = all_points[0][1]
            rng = np.random.default_rng(SEED + 5000 + idx)

            train_hat = simulate_internal_estimates(train_points, lat0, lon0, rng)
            test_hat = simulate_internal_estimates(test_points, lat0, lon0, rng)
            hmm = build_user_hmm(train_hat, lat0, lon0, GRID_SIZE_M)
            burst_windows = build_synthetic_burst_windows(test_points, BURST_LENGTH_SEC)

            vals.append(run_single_trace_for_sensitivity(
                test_points, test_hat, lat0, lon0, burst_windows, hmm,
                rng_seed=SEED + 7000 + idx, budget_B=B, tau=TAU
            ))

        budget_rows.append({
            "B": B,
            "emit_rate": np.nanmean([v["emit_rate"] for v in vals]),
            "budget_used": np.nanmean([v["budget_used"] for v in vals]),
            "median_err": np.nanmean([v["median_err"] for v in vals]),
            "p95_err": np.nanmean([v["p95_err"] for v in vals]),
        })

    # Panel B: vary tau, fixed B=6
    for tau in SENS_TAUS:
        vals = []
        for idx, (trace_id, uid, tr) in enumerate(sens_traces, 1):
            all_points = tr["all_points"]
            test_points = tr["test_points"]
            train_points = tr["train_points"]
            lat0 = all_points[0][0]
            lon0 = all_points[0][1]
            rng = np.random.default_rng(SEED + 9000 + idx)

            train_hat = simulate_internal_estimates(train_points, lat0, lon0, rng)
            test_hat = simulate_internal_estimates(test_points, lat0, lon0, rng)
            hmm = build_user_hmm(train_hat, lat0, lon0, GRID_SIZE_M)
            burst_windows = build_synthetic_burst_windows(test_points, BURST_LENGTH_SEC)

            vals.append(run_single_trace_for_sensitivity(
                test_points, test_hat, lat0, lon0, burst_windows, hmm,
                rng_seed=SEED + 11000 + idx, budget_B=DEFAULT_BUDGET_B, tau=tau
            ))

        tau_rows.append({
            "tau": tau,
            "emit_rate": np.nanmean([v["emit_rate"] for v in vals]),
            "median_err": np.nanmean([v["median_err"] for v in vals]),
            "p95_err": np.nanmean([v["p95_err"] for v in vals]),
            "mean_R_t": np.nanmean([v["mean_R_t"] for v in vals]),
        })

    return pd.DataFrame(budget_rows), pd.DataFrame(tau_rows)


def run_optional_urbannav_experiment():
    rows = load_urbannav_trace(URBANNAV_PATH)
    if rows is None:
        return None

    print("\n[RUN] UrbanNav transfer-case experiment")

    lat0 = rows[0][0]
    lon0 = rows[0][1]
    rng = np.random.default_rng(SEED + 123456)

    # For UrbanNav we treat the whole trace as evaluation trace;
    # HMM training here uses first 50% just to instantiate the attacker state model.
    n = len(rows)
    if n < 40:
        print("[WARN] UrbanNav trace too short, skipping.")
        return None

    k = max(20, int(0.5 * n))
    train_points = [(r[0], r[1], r[2]) for r in rows[:k]]
    test_points = [(r[0], r[1], r[2]) for r in rows[k:]]

    train_hat = simulate_internal_estimates(train_points, lat0, lon0, rng)
    test_hat = simulate_internal_estimates(test_points, lat0, lon0, rng)
    hmm = build_user_hmm(train_hat, lat0, lon0, GRID_SIZE_M)

    burst_windows = build_synthetic_burst_windows(test_points, BURST_LENGTH_SEC)

    static_df, _, _, _ = run_static_gi_release(
        "UrbanNav480", test_points, test_hat, lat0, lon0, rng, DEFAULT_BUDGET_B, burst_windows
    )
    tw_df, _, _, _ = run_twinkle_release(
        "UrbanNav480", test_points, test_hat, lat0, lon0, rng, hmm, burst_windows,
        budget_B=DEFAULT_BUDGET_B, with_auditor=False
    )
    aud_df, _, _, _ = run_twinkle_release(
        "UrbanNav480", test_points, test_hat, lat0, lon0, rng, hmm, burst_windows,
        budget_B=DEFAULT_BUDGET_B, with_auditor=True
    )

    release_df = pd.concat([static_df, tw_df, aud_df], ignore_index=True)
    return release_df


# =========================================================
# Save tables
# =========================================================

def save_pretty_csv(df, path, round_cols=True):
    out = df.copy()
    if round_cols:
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(3)
    out.to_csv(path, index=False)


# =========================================================
# Main
# =========================================================

def main():
    start_time = time.time()

    try:
        print("=" * 80)
        print("TwinkleGPS experiments starting...")
        print("=" * 80)

        # -------------------------
        # GeoLife main only
        # -------------------------
        release_all_df, attack_all_df, geolife_release_summary, attack_summary, example_payloads, flat_traces = run_geolife_main_experiment()

        # -------------------------
        # Save CSVs
        # -------------------------
        save_pretty_csv(
            release_all_df,
            os.path.join(OUTPUT_DIR, "geolife_all_release_records.csv")
        )
        save_pretty_csv(
            attack_all_df,
            os.path.join(OUTPUT_DIR, "geolife_all_attack_records.csv")
        )
        save_pretty_csv(
            geolife_release_summary,
            os.path.join(OUTPUT_DIR, "table_geolife_release_utility.csv")
        )
        save_pretty_csv(
            attack_summary,
            os.path.join(OUTPUT_DIR, "table_geolife_attack_summary.csv")
        )

        # -------------------------
        # Plots: GeoLife utility
        # -------------------------
        print("[PLOT] GeoLife utility CDFs...")
        plot_geolife_utility_cdfs(release_all_df, OUTPUT_DIR)

        # -------------------------
        # Attack plots
        # -------------------------
        print("[PLOT] GeoLife attack figures...")
        plot_overall_attack_bars(attack_all_df, OUTPUT_DIR)
        plot_highrisk_attack_bars(attack_all_df, OUTPUT_DIR)

        release_tw = release_all_df[release_all_df["method"] == "Twinkle"].copy()
        release_aud = release_all_df[release_all_df["method"] == "Twinkle+Auditor"].copy()

        plot_recovery_error_boxplots(attack_all_df, OUTPUT_DIR)
        plot_release_error_boxplots(release_tw, release_aud, OUTPUT_DIR)

        # -------------------------
        # Optional example trajectories
        # Comment this block out too if you want even faster runs
        # -------------------------
        print("[PLOT] Example trajectory figures...")
        for payload in example_payloads:
            trace_id, test_points, tw_df, aud_df, atk_tw, atk_aud = payload
            plot_example_trajectory(
                trace_id,
                test_points,
                tw_df,
                aud_df,
                atk_tw,
                atk_aud,
                OUTPUT_DIR,
                EXAMPLE_MAX_PLOT_POINTS
            )

        # -------------------------
        # Print summaries
        # -------------------------
        print("\n" + "=" * 80)
        print("GeoLife release-side utility summary")
        print("=" * 80)
        print(geolife_release_summary.to_string(index=False))

        print("\n" + "=" * 80)
        print("GeoLife attack summary")
        print("=" * 80)
        print(attack_summary.round(3).to_string(index=False))

        print("\n[INFO] Sensitivity analysis skipped in this fast run.")
        print("[INFO] UrbanNav experiment skipped in this fast run.")

        print("\n" + "=" * 80)
        print(f"All results saved to: {OUTPUT_DIR}")
        print(f"Total elapsed time: {time.time() - start_time:.1f}s")
        print("=" * 80)

    except Exception as e:
        print("\n[FATAL ERROR]")
        print(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()