from __future__ import annotations

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "aat_parameter_distribution_figures_4benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_SEED = 99
INITIAL_A = 1.0
INITIAL_C = 0.0
IMAGE_DPI = 800
SHOW_FIGURES = True


BURGERS_MANUAL_DIR = None
NLSE_MANUAL_DIR = None
AC_MANUAL_DIR = None
WAVE_MANUAL_DIR = None


PROBLEMS = {
    "burgers": {
        "label": "Burgers",
        "checkpoints": [2500, 5000],
        "filename": lambda ev: f"burgers_nmaf_aat_seed{REPRESENTATIVE_SEED}_eval{ev:05d}.pth",
        "manual_dir": lambda: BURGERS_MANUAL_DIR,
        "tick_labels": {0: "0", 2500: "2500", 5000: "5000"},
    },
    "nlse": {
        "label": "NLSE",
        "checkpoints": [3000, 6000, 10000],
        "filename": lambda ev: f"nlse_nmaf_aat_seed{REPRESENTATIVE_SEED}_eval{ev:05d}.pth",
        "manual_dir": lambda: NLSE_MANUAL_DIR,
        "tick_labels": {0: "0", 3000: "3000", 6000: "6000", 10000: "10000"},
    },
    "ac": {
        "label": "Allen–Cahn",
        "checkpoints": [10000, 20000],
        "filename": lambda ev: f"allen_cahn_nmaf_aat_seed{REPRESENTATIVE_SEED}_eval{ev:05d}.pth",
        "manual_dir": lambda: AC_MANUAL_DIR,
        "tick_labels": {0: "0", 10000: "10000", 20000: "20000"},
    },
    "wave": {
        "label": "Wave",
        "checkpoints": [500000, 700000],
        "filename": lambda ev: (
            f"wave_nmaf_aat_seed{REPRESENTATIVE_SEED}_adam0500000.pth"
            if ev == 500000 else
            f"wave_nmaf_aat_seed{REPRESENTATIVE_SEED}_total0700000.pth"
        ),
        "manual_dir": lambda: WAVE_MANUAL_DIR,
        "tick_labels": {0: "0", 500000: "500000", 700000: "700000"},
    },
}


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.0,
    })


def locate_checkpoint(problem: str, checkpoint_value: int) -> Path:
    cfg = PROBLEMS[problem]
    filename = cfg["filename"](checkpoint_value)
    manual_dir = cfg["manual_dir"]()

    if manual_dir is not None:
        path = Path(manual_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Manual checkpoint path does not exist:\n{path}")
        return path

    matches = list(SCRIPT_DIR.rglob(filename))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find checkpoint for {cfg['label']}:\n"
            f"  expected filename: {filename}\n"
            f"  search root      : {SCRIPT_DIR}"
        )
    if len(matches) > 1:
        msg = "\n".join(f"  {m}" for m in matches)
        raise RuntimeError(
            f"Multiple checkpoints named '{filename}' were found:\n{msg}\n"
            f"Set the corresponding *_MANUAL_DIR option to select the formal result."
        )
    return matches[0]


def load_checkpoint(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type in {path}: {type(obj)}")
    return obj["model_state_dict"] if "model_state_dict" in obj else obj


def find_state_key(state: dict, desired_key: str) -> str:
    if desired_key in state:
        return desired_key
    matches = [k for k in state.keys() if k.endswith(desired_key)]
    if len(matches) == 1:
        return matches[0]
    preview = "\n".join(f"  {k}" for k in list(state.keys())[:50])
    raise KeyError(
        f"Could not uniquely locate '{desired_key}'.\nAvailable keys include:\n{preview}"
    )


def extract_a_c(path: Path):
    state = load_checkpoint(path)
    a_key = find_state_key(state, "activation.alpha")
    c_key = find_state_key(state, "activation.beta")

    a = state[a_key].detach().cpu().numpy().astype(np.float64).reshape(-1)
    c = state[c_key].detach().cpu().numpy().astype(np.float64).reshape(-1)

    if a.shape != c.shape:
        raise ValueError(f"a/c shape mismatch in {path}: a={a.shape}, c={c.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(c)):
        raise ValueError(f"Non-finite adaptive parameters found in {path}")
    return a, c


def build_problem_arrays(problem: str):
    checkpoints = list(PROBLEMS[problem]["checkpoints"])
    first_path = locate_checkpoint(problem, checkpoints[0])
    first_a, first_c = extract_a_c(first_path)
    n = first_a.size

    arrays = {
        0: {
            "a": np.full(n, INITIAL_A, dtype=np.float64),
            "c": np.full(n, INITIAL_C, dtype=np.float64),
            "path": "analytical initialization",
        }
    }

    for cp in checkpoints:
        path = locate_checkpoint(problem, cp)
        a, c = extract_a_c(path)
        arrays[cp] = {"a": a, "c": c, "path": str(path)}

    return arrays


def save_show_close(fig, filename: str):
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    print(f"Saved: {out}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def boxplot_with_points(ax, arrays, checkpoints, parameter: str, tick_labels):
    values = [arrays[cp][parameter] for cp in checkpoints]
    positions = np.arange(len(checkpoints), dtype=float)

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=False,
        showfliers=False,
        medianprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    for i, val in enumerate(values):
        offsets = np.array([0.0]) if val.size == 1 else np.linspace(-0.14, 0.14, val.size)
        ax.scatter(
            np.full(val.size, positions[i]) + offsets,
            val,
            s=8,
            alpha=0.45,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)

    if parameter == "a":
        ax.set_ylabel(r"Scaling parameter $a_j$")
        ax.axhline(INITIAL_A, linestyle="--", linewidth=1.0)
    else:
        ax.set_ylabel(r"Shifting parameter $c_j$")
        ax.axhline(INITIAL_C, linestyle="--", linewidth=1.0)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45)


def plot_checkpoint_distributions(problem: str, arrays: dict):
    cfg = PROBLEMS[problem]
    checkpoints = [0] + list(cfg["checkpoints"])
    tick_labels = [cfg["tick_labels"][cp] for cp in checkpoints]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    boxplot_with_points(axes[0], arrays, checkpoints, "a", tick_labels)
    boxplot_with_points(axes[1], arrays, checkpoints, "c", tick_labels)
    axes[0].set_xlabel("Iteration")
    axes[1].set_xlabel("Iteration")


    fig.text(
        0.5,
        0.015,
        f"{cfg['label']}: checkpoint-wise AAT parameter distributions",
        ha="center",
        va="bottom",
        fontsize=14,
    )

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))

    save_show_close(fig, f"{problem}_aat_parameter_distributions_iteration.png")


def main():
    configure_plot_style()

    print("=" * 108)
    print("Generating checkpoint-wise AAT parameter distribution figures for four benchmarks")
    print("=" * 108)
    print(f"Script root : {SCRIPT_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Seed        : {REPRESENTATIVE_SEED}")
    print("=" * 108)

    for problem in ["burgers", "nlse", "ac", "wave"]:
        print(f"\n[{PROBLEMS[problem]['label']}] locating checkpoints...")
        arrays = build_problem_arrays(problem)
        for cp in PROBLEMS[problem]["checkpoints"]:
            print(f"  {cp}: {arrays[cp]['path']}")
        plot_checkpoint_distributions(problem, arrays)

    print("\nCompleted.")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
