import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


YMIN = 1.0e-5
YMAX = 1.0e3


excel_path = "loss.xlsx"
df = pd.read_excel(excel_path)

df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

if df.empty:
    raise RuntimeError("loss.xlsx contains no valid data.")

iterations = np.arange(1, len(df) + 1)


plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots(figsize=(7, 5))


COLUMN_RULES = [
    {
        "aliases": ["tanh"],
        "label": "tanh",
        "color": "#1f77b4",
        "zorder": 5,
    },
    {
        "aliases": ["xtanhx", "xtanh", "x_tanh_x", "x-tanh-x"],
        "label": r"$x\tanh(x)$",
        "color": "#2ca02c",
        "zorder": 6,
    },
    {
        "aliases": ["gaussian", "gaussion", "gauss"],
        "label": "Hermite-Gaussian",
        "color": "#9467bd",
        "zorder": 4,
    },
    {
        "aliases": ["nmaf", "ihaf"],
        "label": "NMAF",
        "color": "tab:red",
        "zorder": 7,
    },
]


def normalize_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(aliases):
    normalized = {normalize_name(col): col for col in df.columns}

    for alias in aliases:
        key = normalize_name(alias)

        if key in normalized:
            return normalized[key]

    return None


found_methods = 0

for rule in COLUMN_RULES:
    col = find_column(rule["aliases"])

    if col is None:
        print(f"Skipped: column not found for {rule['label']}")
        continue

    values = pd.to_numeric(
        df[col],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(values) & (values > 0.0)

    if not np.any(valid):
        print(f"Skipped: no positive finite values for {rule['label']}")
        continue

    ax.semilogy(
        iterations[valid],
        values[valid],
        linewidth=1.5,
        color=rule["color"],
        label=rule["label"],
        zorder=rule["zorder"],
    )

    found_methods += 1

if found_methods == 0:
    raise RuntimeError(
        "No valid loss columns found. "
        f"Current Excel columns: {list(df.columns)}"
    )


ax.yaxis.set_major_locator(
    LogLocator(
        base=10.0,
        numticks=20,
    )
)

ax.yaxis.set_minor_locator(
    LogLocator(
        base=10.0,
        subs=np.arange(2, 10) * 0.1,
        numticks=100,
    )
)

ax.yaxis.set_minor_formatter(
    NullFormatter()
)

ax.set_xlabel(
    "Iteration",
    fontsize=14,
)

ax.set_ylabel(
    "Total loss",
    fontsize=14,
)

ax.set_xlim(
    1,
    len(df),
)


if YMIN is not None and YMAX is not None:
    if YMIN <= 0 or YMAX <= 0:
        raise ValueError(
            "For a logarithmic y-axis, YMIN and YMAX must be > 0."
        )

    if YMIN >= YMAX:
        raise ValueError(
            "YMIN must be smaller than YMAX."
        )

    ax.set_ylim(
        YMIN,
        YMAX,
    )

ax.legend(
    loc="upper right",
    fontsize=10,
)

ax.grid(
    True,
    which="major",
    linestyle="-",
    alpha=0.6,
)

ax.grid(
    True,
    which="minor",
    linestyle="--",
    alpha=0.3,
)

plt.tight_layout()


output_path = "NLSE_xtanh_nmaf_gaussian_loss.png"

plt.savefig(
    output_path,
    dpi=800,
    format="png",
    bbox_inches="tight",
)

print(f"Saved: {output_path}")
plt.show()
