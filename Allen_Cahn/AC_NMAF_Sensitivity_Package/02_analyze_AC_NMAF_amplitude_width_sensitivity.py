from __future__ import annotations

import csv
import json
import os
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "ac_nmaf_sensitivity_results"
REFERENCE_FILE = SCRIPT_DIR / "AC.mat"
OUTPUT_DIR = RESULT_ROOT / "analysis_seed99_override_AB_loss_shaded_v6"


SEED99_NMAF_CHECKPOINT = Path(
    r"C:\Users\FANFAN\Desktop\review\Allen\allen_cahn_multiseed_results\nmaf\seed_99\allen_cahn_nmaf_seed99_eval20000.pth"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [2026, 99, 1234]
VALUES = [0.5, 0.75, 1.0, 1.5, 2.0]
HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 128
DTYPE = torch.float64
DEVICE = torch.device("cpu")

FIGSIZE = (8.7, 5.5)
LOSS_FIGSIZE = (7.0, 5.0)
IMAGE_DPI = 800
SHOW_FIGURES = True
FONT_FAMILY = "Times New Roman"
BAR_COLOR = "#C0B3D0"
EDGE_COLOR = "black"
BAR_WIDTH = 0.80
CAPSIZE = 5

SEED_STYLE = {
    2026: {"marker": "o", "offset": -0.12, "label": "Seed 2026"},
    99:   {"marker": "s", "offset": 0.00,  "label": "Seed 99"},
    1234: {"marker": "^", "offset": 0.12,  "label": "Seed 1234"},
}

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = FONT_FAMILY
plt.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
})


def fmt_value(v: float) -> str:
    return (f"{v:g}").replace(".", "p")


def case_name(a: float, sigma: float) -> str:
    return f"A_{fmt_value(a)}_sigma_{fmt_value(sigma)}"


class NMAFActivation(nn.Module):
    def __init__(self, amplitude: float, sigma: float):
        super().__init__()
        self.amplitude = float(amplitude)
        self.sigma = float(sigma)

    def forward(self, x):
        return x * torch.tanh(x) - self.amplitude * x * torch.exp(
            -(x ** 2) / (2.0 * self.sigma ** 2)
        )


class NMAFPINN(nn.Module):
    def __init__(self, amplitude: float, sigma: float):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.activation = NMAFActivation(amplitude, sigma)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        z = self.activation(self.input_layer(z))
        for layer in self.hidden_layers:
            z = self.activation(layer(z))
        return self.output_layer(z)


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {REFERENCE_FILE}. Place AC.mat beside the analysis script."
        )
    data = scipy.io.loadmat(REFERENCE_FILE)
    for key in ("x", "tt", "uu"):
        if key not in data:
            raise KeyError(f"'{key}' not found in {REFERENCE_FILE}")

    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["tt"], dtype=np.float64).reshape(-1)
    uu = np.asarray(data["uu"], dtype=np.float64)


    if uu.shape == (x.size, t.size):
        u_exact = uu
    elif uu.shape == (t.size, x.size):
        u_exact = uu.T
    else:
        raise ValueError(
            f"Unexpected uu shape {uu.shape}; expected {(x.size, t.size)} "
            f"or {(t.size, x.size)}."
        )
    return x, t, u_exact


def predict_full_grid(model, x, t, batch_size=65536):

    X, T = np.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    pred = np.empty((xf.shape[0], 1), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, xf.shape[0], batch_size):
            stop = min(start + batch_size, xf.shape[0])
            xb = torch.tensor(xf[start:stop], dtype=DTYPE, device=DEVICE)
            tb = torch.tensor(tf[start:stop], dtype=DTYPE, device=DEVICE)
            pred[start:stop] = model(xb, tb).cpu().numpy()
    return pred.reshape(x.size, t.size)


def is_seed99_override(a, sigma, seed):
    return a == 1.0 and sigma == 1.0 and seed == 99


class OriginalFixedNMAF(nn.Module):

    def forward(self, z):
        return -z * (torch.exp(-0.5 * z ** 2) - torch.tanh(z))


def load_seed99_checkpoint():
    if not SEED99_NMAF_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing override checkpoint: {SEED99_NMAF_CHECKPOINT}")
    ckpt = torch.load(SEED99_NMAF_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if ckpt.get("seed") != 99 or ckpt.get("method") != "nmaf":
        raise ValueError("Override checkpoint must be fixed NMAF, seed 99.")
    return ckpt


def evaluate_case(a: float, sigma: float, seed: int, x, t, u_exact):
    c = case_name(a, sigma)
    seed_dir = RESULT_ROOT / c / f"seed_{seed}"
    override = is_seed99_override(a, sigma, seed)
    ckpt_path = SEED99_NMAF_CHECKPOINT if override else seed_dir / "final_checkpoint.pth"
    summary_path = seed_dir / "training_summary.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not override and not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    ckpt = load_seed99_checkpoint() if override else torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = NMAFPINN(a, sigma).double().to(DEVICE)
    if override:
        model.activation = OriginalFixedNMAF()
        print(f"Using override: {ckpt_path}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    pred = predict_full_grid(model, x, t)

    diff = pred - u_exact
    rel_l2 = np.linalg.norm(diff.ravel(), 2) / np.linalg.norm(u_exact.ravel(), 2)
    max_abs = np.max(np.abs(diff))

    if override:

        summary = {
            "final_total_loss": ckpt["latest_metrics"]["total_loss"],
            "training_time_seconds": ckpt["elapsed_seconds"],
            "actual_lbfgs_eval": ckpt["actual_lbfgs_eval"],
        }
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "case": c,
        "checkpoint_path": str(ckpt_path),
        "seed": int(seed),
        "amplitude": float(a),
        "sigma": float(sigma),
        "final_total_loss": float(summary["final_total_loss"]),
        "relative_L2_error": float(rel_l2),
        "maximum_absolute_error": float(max_abs),
        "training_time_seconds": float(summary["training_time_seconds"]),
        "actual_lbfgs_eval": int(summary["actual_lbfgs_eval"]),
    }


def sample_mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def branch_rows(all_rows, branch: str):
    if branch == "amplitude":
        return [r for r in all_rows if np.isclose(r["sigma"], 1.0)]
    if branch == "width":
        return [r for r in all_rows if np.isclose(r["amplitude"], 1.0)]
    raise ValueError(branch)


def aggregate(all_rows, branch: str):
    rows = branch_rows(all_rows, branch)
    metric_names = [
        "final_total_loss", "relative_L2_error", "maximum_absolute_error",
        "training_time_seconds", "actual_lbfgs_eval",
    ]
    out = []
    for value in VALUES:
        selected = [
            r for r in rows
            if np.isclose(r["amplitude" if branch == "amplitude" else "sigma"], value)
        ]
        if len(selected) != len(SEEDS):
            raise RuntimeError(f"Expected {len(SEEDS)} rows for {branch}={value}; got {len(selected)}")
        row = {"branch": branch, "parameter_value": value}
        for metric in metric_names:
            mean, sd = sample_mean_std([r[metric] for r in selected])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        out.append(row)
    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sci_axes(ax):
    ax.tick_params(direction="in", length=5, width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_metric(all_rows, aggregates, metric, ylabel, stem, log_y=False):

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    panel_defs = [
        ("amplitude", r"Correction amplitude $A$"),
        ("width", r"Correction width $B$"),
    ]

    for ax, (branch, xlabel) in zip(axes, panel_defs):
        stats = [r for r in aggregates if r["branch"] == branch]
        means = np.array([r[f"{metric}_mean"] for r in stats])
        sds = np.array([r[f"{metric}_sd"] for r in stats])
        pos = np.arange(len(VALUES), dtype=float)


        yerr = np.vstack([np.minimum(sds, 0.95 * means), sds]) if log_y else sds
        bars = ax.bar(
            pos, means, width=BAR_WIDTH,
            yerr=yerr, capsize=CAPSIZE,
            alpha=0.78,
            color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.7,
            zorder=2,
        )

        rows = branch_rows(all_rows, branch)
        for seed in SEEDS:
            st = SEED_STYLE[seed]
            ys = []
            for v in VALUES:
                matches = [
                    r for r in rows
                    if r["seed"] == seed
                    and np.isclose(r["amplitude" if branch == "amplitude" else "sigma"], v)
                ]
                ys.append(matches[0][metric])
            ax.scatter(
                pos + st["offset"], ys,
                marker=st["marker"], s=40,
                edgecolors="black", linewidths=0.6,
                zorder=4, label=st["label"],
            )


        original_idx = VALUES.index(1.0)
        ax.axvline(original_idx, linestyle="--", linewidth=1.0, color="0.35", zorder=1)

        ax.set_xticks(pos)
        ax.set_xticklabels([f"{v:g}" for v in VALUES])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            )
            ax.yaxis.set_minor_formatter(NullFormatter())


        ax.grid(True, axis="y", which="both", alpha=0.35)
        ax.set_axisbelow(True)


        for rect, value in zip(bars, means):
            x_text = rect.get_x() + rect.get_width() * 0.90
            y_text = value * 0.90
            label_text = f"{value:.2e}" if (log_y or abs(value) < 1.0e-2 or abs(value) >= 1.0e3) else f"{value:.2f}"
            ax.annotate(
                label_text,
                xy=(x_text, y_text),
                xytext=(0, 0),
                textcoords="offset points",
                ha="right", va="top", fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
                zorder=5,
            )


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, frameon=True, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight", pad_inches=0.03)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {out}")


def plot_loss_histories_original_context():


    for branch in ("amplitude", "width"):


        fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)
        xmax = 1
        for value in VALUES:
            histories = []
            for seed in SEEDS:
                a, sig = (value, 1.0) if branch == "amplitude" else (1.0, value)
                path = RESULT_ROOT / case_name(a, sig) / f"seed_{seed}" / "loss_history.txt"
                if is_seed99_override(a, sig, seed):
                    arr = np.asarray(load_seed99_checkpoint()["loss_history"], dtype=np.float64)
                else:
                    arr = np.loadtxt(path)
                histories.append(arr[:, :2])


            x = histories[0][:, 0]
            for history in histories[1:]:
                x = np.intersect1d(x, history[:, 0])
            if x.size == 0:
                raise ValueError(f"No common iterations for {branch}={value}")
            aligned = []
            for history in histories:
                if np.any(np.diff(history[:, 0]) <= 0):
                    raise ValueError("History iterations must be strictly increasing.")
                aligned.append(history[np.searchsorted(history[:, 0], x), 1])
            y = np.vstack(aligned)
            mean = np.mean(y, axis=0)
            sd = np.std(y, axis=0, ddof=1)
            valid = (np.all(np.isfinite(y) & (y > 0.0), axis=0)
                     & np.isfinite(x))
            if np.any(valid):
                line, = ax.semilogy(
                    x[valid], mean[valid], linewidth=1.5,
                    label=f"{'A' if branch == 'amplitude' else 'B'}={value:g}",
                )


                floor = 1e-6 if branch == "amplitude" else float(np.min(y[:, valid])) * 0.1
                ax.fill_between(
                    x, np.maximum(mean - sd, floor), mean + sd,
                    where=valid, color=line.get_color(), alpha=0.20,
                    linewidth=0, zorder=1,
                )
                xmax = max(xmax, float(np.max(x[valid])))

        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Total loss", fontsize=14)
        ax.set_xlim(1, xmax)
        if branch == "amplitude":
            ax.set_ylim(1e-6, 1e1)

        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle="--", alpha=0.3)

        fig.tight_layout()
        out = OUTPUT_DIR / f"diagnostic_{branch}_mean_loss_histories.png"
        fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved: {out}")


def main():
    x, t, u_exact = load_reference()

    all_rows = []


    unique_cases = [(v, 1.0) for v in VALUES] + [(1.0, v) for v in VALUES if v != 1.0]
    for a, s in unique_cases:
        for seed in SEEDS:
            row = evaluate_case(a, s, seed, x, t, u_exact)
            all_rows.append(row)
            print(
                f"A={a:g}, sigma={s:g}, seed={seed}: "
                f"loss={row['final_total_loss']:.3e}, "
                f"L2={row['relative_L2_error']:.3e}, "
                f"max={row['maximum_absolute_error']:.3e}"
            )

    write_csv(OUTPUT_DIR / "seedwise_metrics.csv", all_rows)

    agg = aggregate(all_rows, "amplitude") + aggregate(all_rows, "width")
    write_csv(OUTPUT_DIR / "mean_sd_metrics.csv", agg)

    plot_metric(
        all_rows, agg,
        metric="relative_L2_error",
        ylabel=r"Relative $L_2$ error",
        stem="Fig_sensitivity_relative_L2_error",
        log_y=True,
    )
    plot_metric(
        all_rows, agg,
        metric="maximum_absolute_error",
        ylabel="Maximum absolute error",
        stem="Fig_sensitivity_maximum_absolute_error",
        log_y=True,
    )
    plot_metric(
        all_rows, agg,
        metric="final_total_loss",
        ylabel="Final total loss",
        stem="Fig_sensitivity_final_total_loss",
        log_y=True,
    )
    plot_metric(
        all_rows, agg,
        metric="training_time_seconds",
        ylabel="Wall-clock training time (s)",
        stem="Fig_sensitivity_training_time",
        log_y=False,
    )


    lines = []
    for branch in ("amplitude", "width"):
        lines.append(branch.upper())
        lines.append("value | final loss (mean±SD) | relative L2 (mean±SD) | max abs (mean±SD)")
        for r in [q for q in agg if q["branch"] == branch]:
            lines.append(
                f"{r['parameter_value']:g} | "
                f"{r['final_total_loss_mean']:.4e} ± {r['final_total_loss_sd']:.4e} | "
                f"{r['relative_L2_error_mean']:.4e} ± {r['relative_L2_error_sd']:.4e} | "
                f"{r['maximum_absolute_error_mean']:.4e} ± {r['maximum_absolute_error_sd']:.4e}"
            )
        lines.append("")
    (OUTPUT_DIR / "manuscript_ready_summary.txt").write_text("\n".join(lines), encoding="utf-8")


    plot_loss_histories_original_context()

    print("\nAnalysis complete.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
