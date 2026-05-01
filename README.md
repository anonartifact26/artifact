
# TwinkleGPS

This repository contains the experimental code for the paper on **TwinkleGPS**, including **main real-device deployment**, **GeoLife real-trace replay**, **replay utility evaluation**, **sequential trajectory recovery**, and **sensitivity analysis**.

The experiments in the paper are organized in the following order:

1. **Main real-device deployment**
2. **GeoLife real-trace replay**
3. **Replay utility**
4. **Sequential trajectory recovery**
5. **Sensitivity**


## Quick start

1. Download the **GeoLife GPS Trajectories** dataset from the official source.
2. Extract it locally so that the directory structure looks like:

    ```text
    GeoLife/
    └── Data/
        ├── 000/
        │   └── Trajectory/
        │       ├── *.plt
        │       └── ...
        ├── 001/
        │   └── Trajectory/
        │       └── ...
        └── ...
    ```

3. In the relevant scripts, replace:

    ```python
    "path/to/Geolife/Data"
    ```

    with your local GeoLife path.

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Run the desired experiment:

    ```bash
    python "Main real-device deployment code"
    python "GeoLife real-trace replay.py"
    python "Replay utility.py"
    python "Sequential trajectory recovery.py"
    python "Sensitivity.py"
    ```

---

## Repository overview

This repository is intended to support the main experimental results reported in the paper.

### Currently included

- Main real-device deployment code
- GeoLife-based real-trace replay
- Release-side utility evaluation
- Sequential trajectory recovery under a user-specific attacker
- Sensitivity analysis for key Twinkle+Auditor parameters

### Current release status

- **Experiment 1** is documented in this README and corresponds to the local code and data organization used in this project.
- **Experiments 2--5** are implemented in the current repository and can be run after local path configuration.

---

## Repository structure

A typical repository layout is:

```text
TwinkleGPS/
├── README.md
├── requirements.txt
├── .gitignore
├── Main real-device deployment code
├── GeoLife real-trace replay.py
├── Replay utility.py
├── Sequential trajectory recovery.py
├── Sensitivity.py
├── data/
│   └── experiment/
│       └── data/
├── outputs/
├── figures_exp3_geolife/
└── sensitivity_results_paper_consistent/
```

The GeoLife dataset is **not included** in this repository.

---

## Installation

We recommend using **Python 3.10 or later**.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The current code mainly depends on:

- `numpy`
- `pandas`
- `matplotlib`

Depending on the exact local environment and script version, additional standard scientific Python packages may also be required.

---

## Dataset

This repository uses two types of data:

1. **Real-device deployment data** for **Experiment 1**
2. **GeoLife GPS Trajectories** for **Experiments 2--5**

### Real-device deployment data

The local data directory for the main real-device deployment experiment is:

```text
data/experiment/data
```

This directory is intended to store the real-device walking logs used in **Experiment 1**.

A typical structure may look like:

```text
data/
└── experiment/
    └── data/
        ├── order1/
        │   ├── baseline.xlsx
        │   ├── burst.xlsx
        │   └── gps.xlsx
        ├── order2/
        │   ├── baseline.xlsx
        │   ├── burst.xlsx
        │   └── gps.xlsx
        └── order3/
            ├── baseline.xlsx
            ├── burst.xlsx
            └── gps.xlsx
```

### GeoLife dataset

The GeoLife dataset is **not included** in this repository. Please download it from the official source and place it locally before running the replay-based code.

Expected directory structure:

```text
GeoLife/
└── Data/
    ├── 000/
    │   └── Trajectory/
    │       ├── *.plt
    │       └── ...
    ├── 001/
    │   └── Trajectory/
    │       └── ...
    └── ...
```

In the provided scripts, the GeoLife path should point to:

```python
"path/to/Geolife/Data"
```

That is, the path should refer directly to the `Data` directory.

---

## Experiment 1: Main real-device deployment

This experiment corresponds to the **main on-device evaluation** of TwinkleGPS under real walking traces collected from actual mobile devices.

### Code and data names used in this project

The local code file for this experiment is:

```text
Main real-device deployment code
```

The local data directory for this experiment is:

```text
data/experiment/data
```

### Purpose

The goal of this experiment is to evaluate how TwinkleGPS behaves in a real deployment setting rather than in replay only. In particular, this experiment is intended to compare utility and energy behavior under practical device-side execution.

### Compared profiles

The real-device experiment uses the following profiles:

- **baseline**
- **burst**
- **gps**

A typical interpretation is:

- **baseline**: the baseline Twinkle-style release configuration during normal operation
- **burst**: a higher-frequency release configuration during public-intent or burst periods
- **gps**: a GPS-based reference run used as the comparison path or ground-truth-like trajectory reference

### Experimental organization

The real-device deployment is organized by repeated run orders, typically such as:

- `order1`
- `order2`
- `order3`

For each order, the experiment records outputs for the different profiles listed above.

### Input data format

The experiment is designed around structured Excel logs collected during real-device runs. Typical files are organized under directories such as:

```text
data/
└── experiment/
    └── data/
        ├── order1/
        │   ├── baseline.xlsx
        │   ├── burst.xlsx
        │   └── gps.xlsx
        ├── order2/
        │   ├── baseline.xlsx
        │   ├── burst.xlsx
        │   └── gps.xlsx
        └── order3/
            ├── baseline.xlsx
            ├── burst.xlsx
            └── gps.xlsx
```

Some setups may alternatively store each profile in a subdirectory containing one or more Excel files. In that case, the analysis code typically selects the main Excel file from the corresponding folder.

### Typical logged fields

The real-device logs may include columns such as:

- `timestamp`
- `elapsed_ms`
- `event`
- `mode`
- `intent`
- `raw_provider`
- `raw_lat`
- `raw_lon`
- `raw_accuracy_m`
- `released_lat`
- `released_lon`
- `release_offset_m`
- `epsilon_used`
- `release_triggered`
- `raw_update_count`
- `release_count`
- `gps_reference_count`
- `battery_pct`
- `is_charging`
- `charge_plug`
- `device`
- `android_version`
- `note`

These fields support downstream analysis of release behavior, path error, and battery consumption.

### Main evaluation dimensions

This experiment is intended to support the following real-device analyses:

- trajectory utility relative to the GPS reference
- release behavior under different profiles
- battery consumption during real runs
- qualitative and quantitative comparison between baseline and burst behavior

### Typical metrics

The real-device deployment analysis typically focuses on:

- path error relative to the GPS reference path
- per-release spatial error
- start battery level
- end battery level
- total battery drop
- experiment duration
- estimated battery drain rate per hour

### Expected outputs

Typical outputs for the real-device deployment may include:

- summary CSV tables
- per-order profile comparisons
- path visualizations
- battery comparison plots
- PNG and PDF figures for reporting

If an analysis script is used for this experiment, outputs are typically written to a directory such as:

```text
outputs/walking/
```

### How to run

If the code file is named exactly as below in your local project, run:

```bash
python "Main real-device deployment code"
```

If the script requires local path variables, make sure they point to:

```text
data/experiment/data
```

### Notes

This experiment appears first in the paper because it reflects the main deployment setting.

---

## Experiment 2: GeoLife real-trace replay

This experiment replays trajectories from the **GeoLife GPS Trajectories** dataset and applies the TwinkleGPS release logic under simulated device-side position estimates.

### Purpose

The goal of this experiment is to evaluate the release mechanism on real mobility traces while preserving the paper-aligned scheduling, budgeting, and privacy-control logic.

### Main mechanisms

The replay pipeline includes the following mechanisms:

- **Static GI**
- **Twinkle**
- **Twinkle+Auditor**

### Replay assumptions

The replay code simulates an internal device-side estimate from the underlying trajectory using noise and bias models. Synthetic public-intent burst windows are inserted at fixed relative positions along each trajectory.

### GeoLife preprocessing protocol

Across the replay-based experiments, the code follows the paper-aligned preprocessing rules:

- retain trajectories with at least **80 samples**
- truncate retained trajectories to at most **1200 samples**
- keep at most **3 trajectories per user**, where applicable

The replay utility code additionally samples a target number of trajectories for tractability.

### Burst schedule

The replay setup places synthetic burst windows at relative trajectory positions:

- **25%**
- **55%**
- **80%**

Each burst window has duration:

- **60 seconds**

### Default paper-aligned parameters

The main replay parameters appearing in the current code are:

- Static GI interval: **10 s**
- Static GI epsilon: **0.06**
- Twinkle baseline interval: **30 s**
- Twinkle burst interval: **2 s**
- Twinkle baseline epsilon: **0.03**
- Twinkle burst epsilon: **0.12**
- Total privacy budget: **6.0**

### How to run

Set the GeoLife path in the corresponding scripts to:

```python
GEOLIFE_ROOT = "path/to/Geolife/Data"
```

or

```python
DATA_ROOT = "path/to/Geolife/Data"
```

depending on the script.

Then run the experiment scripts described below.

---

## Experiment 3: Replay utility

This experiment evaluates the **release-side utility** of replayed trajectories.

### Purpose

The main objective is to compare release distortion under:

- **Static GI**
- **Twinkle**
- **Twinkle+Auditor**

The code also studies how replay utility changes across simulated device capability tiers and under capability-aware epsilon scaling.

### Simulated device capability tiers

The replay utility code simulates the following internal-estimate tiers:

- **L1-only**
- **L1+L5**
- **L1+L5+ADR**

These capability tiers are used only in the replay simulation and do not correspond to raw device logs in this repository.

### Utility metrics

The code reports release-time position errors in meters, including:

- all release times
- burst-window-restricted release times
- fixed-epsilon comparisons across capability tiers
- capability-aware epsilon scaling comparisons

Summary statistics include:

- number of emitted points
- median error
- 95th percentile error
- mean error

### Outputs

Typical outputs produced by **Replay utility** include:

- `exp3_cdf_l1_all_geolife.png`
- `exp3_cdf_l1_burst_geolife.png`
- `exp3_cdf_fixed_eps_staticGI_geolife.png`
- `exp3_cdf_fixed_eps_twinkleAuditor_geolife.png`
- `exp3_cdf_eps_scaled_across_devices_geolife.png`
- `exp3_geolife_paper_stats.csv`
- `exp3_geolife_paper_stats.json`
- `exp3_geolife_meta.csv`
- `exp3_geolife_progress_meta.csv`

### Script behavior

The current utility script:

- loads a sampled subset of valid GeoLife trajectories
- converts trajectories into a local ENU frame
- simulates internal device estimates
- runs Static GI, Twinkle, and Twinkle+Auditor
- aggregates release-time error statistics
- saves figures and machine-readable summary files

### How to run

Make sure the GeoLife path is set correctly in the script:

```python
GEOLIFE_ROOT = "path/to/Geolife/Data"
OUT_DIR = "figures_exp3_geolife"
```

Then run:

```bash
python "Replay utility.py"
```

---

## Experiment 4: Sequential trajectory recovery

This experiment evaluates a **sequential attacker** against released locations.

### Purpose

The goal is to compare:

- **Twinkle**
- **Twinkle+Auditor**

under a trajectory recovery attack based on user-specific training data.

### Attacker model

The current code implements a user-specific HMM-style sequential recovery pipeline.

For each valid trajectory:

1. the GeoLife trace is loaded and cleaned
2. a train/test split is created
3. device-side internal estimates are simulated
4. sanitized releases are generated
5. a sequential attacker performs recovery on emitted timestamps only

### Preprocessing and train/test protocol

The sequential recovery pipeline applies additional cleaning steps beyond the utility-only replay script, including:

- timestamp parsing
- removal of invalid or degenerate transitions
- filtering of implausibly fast movement
- train/test split for attacker construction

The paper-aligned split configuration in the current code includes:

- train ratio: **0.75**
- minimum training points: **40**
- minimum test points: **20**

### Release mechanisms

The code compares:

- **Static GI**
- **Twinkle**
- **Twinkle+Auditor**

For the attack comparison itself, the main focus is on:

- **Twinkle**
- **Twinkle+Auditor**

### Attack evaluation metrics

The sequential recovery evaluation reports:

- **Exact (%)**
- **Hit@100 m (%)**
- **Hit@300 m (%)**
- **Hit@500 m (%)**
- median reconstruction error
- IQR reconstruction error

It also reports results for:

- all emitted timestamps
- high-risk emitted timestamps

### Outputs

Typical outputs produced by **Sequential trajectory recovery** include:

- `geolife_all_release_records.csv`
- `geolife_all_attack_records.csv`
- `table_geolife_release_utility.csv`
- `table_geolife_attack_summary.csv`
- `fig1_overall_attack_success.png`
- `fig2_highrisk_attack_success.png`
- `fig3_recovery_error_boxplots.png`
- `fig4_release_error_boxplot.png`

Optional example trajectory visualizations may also be produced, for example:

- `fig10_example_trajectory_<trace_id>.png`

### Optional UrbanNav support

The script also contains optional support for an UrbanNav transfer-style case study. If unused, this path can remain empty:

```python
URBANNAV_PATH = ""
```

### How to run

Set the paths in the script:

```python
DATA_ROOT = "path/to/Geolife/Data"
OUTPUT_DIR = "outputs"
```

Then run:

```bash
python "Sequential trajectory recovery.py"
```

### Notes

The default main routine executes the GeoLife main experiment and saves the core outputs.

In the current default run:

- sensitivity analysis is defined but not automatically executed there
- UrbanNav evaluation is optional and not required
- example trajectory plots may be generated when suitable traces exist

---

## Experiment 5: Sensitivity

This experiment studies how **Twinkle+Auditor** behaves when key parameters are varied.

### Purpose

The sensitivity study focuses on the utility-risk-behavior tradeoff under parameter changes.

The current code sweeps:

- **budget_B**
- **risk_tau**

### Parameter grids

The default grids in the current code are:

- `budget_B`: `2.0, 4.0, 6.0, 8.0, 10.0`
- `risk_tau`: `0.4, 0.5, 0.6, 0.7, 0.8`

### Setting

The current sensitivity study is restricted to the:

- **L1-only** setting

This is explicitly enforced in the code.

### Metrics

For each parameter value, the code aggregates per-trajectory summaries including:

- median error at all emissions
- p95 error at all emissions
- mean error at all emissions
- median error in burst windows
- p95 error in burst windows
- mean error in burst windows
- composite risk
- route risk
- place risk
- emission rate
- final budget consumed
- average alpha
- average effective epsilon

### Outputs

Typical outputs produced by **Sensitivity** include:

For `budget_B`:

- `sensitivity_budget_B.csv`
- `sensitivity_budget_B_summary.png`
- `sensitivity_budget_B_utility_ribbon.png`
- `sensitivity_budget_B_risk_area.png`
- `sensitivity_budget_B_behavior_bar.png`
- `sensitivity_budget_B_tradeoff_bubble.png`
- `sensitivity_budget_B_heatmap.png`

For `risk_tau`:

- `sensitivity_risk_tau.csv`
- `sensitivity_risk_tau_summary.png`
- `sensitivity_risk_tau_utility_ribbon.png`
- `sensitivity_risk_tau_risk_area.png`
- `sensitivity_risk_tau_behavior_bar.png`
- `sensitivity_risk_tau_tradeoff_bubble.png`
- `sensitivity_risk_tau_heatmap.png`

### How to run

Set the path in the script:

```python
geolife_root = "path/to/Geolife/Data"
out_dir = "sensitivity_results_paper_consistent"
```

Then run:

```bash
python "Sensitivity.py"
```

---

## Outputs

Depending on the experiment, the scripts generate figures and tables under directories such as:

- `outputs/`
- `figures_exp3_geolife/`
- `sensitivity_results_paper_consistent/`

These generated outputs are typically not required to be tracked in GitHub unless you want to publish selected example results.

---

## Data availability

The GeoLife dataset is **not included** in this repository.

Please download it separately from the official source and place it on your local machine. This repository only contains the code used to process and evaluate the traces.

The local real-device deployment data for **Experiment 1** is organized under:

```text
data/experiment/data
```

If you plan to share this repository publicly, you may choose whether to include or exclude that directory depending on your data release policy.

---

## Reproducibility notes

The current scripts use fixed random seeds in multiple parts of the pipeline for reproducibility.

However, exact numerical values may still vary slightly depending on:

- Python version
- NumPy version
- pandas version
- matplotlib version
- operating system
- local file ordering or sampling conditions if the dataset layout changes

For **Experiment 1**, exact reproducibility additionally depends on the availability of real-device logs, device configuration, and deployment protocol details.

---

## Minimal usage checklist

Before running the experiments:

1. download the GeoLife dataset
2. extract it locally
3. make sure the directory looks like `.../GeoLife/Data/<user_id>/Trajectory/*.plt`
4. update the path strings in the scripts from:

    ```python
    "path/to/Geolife/Data"
    ```

    to your actual local path
5. install the dependencies in `requirements.txt`
6. run the desired experiment script

If you plan to run **Experiment 1**, also make sure your local real-device data is placed under:

```text
data/experiment/data
```

and that the corresponding script file is named:

```text
Main real-device deployment code
```

---

## Citation

If you use this code, please cite the corresponding paper.

> Citation details will be added after the paper metadata is finalized.

---
