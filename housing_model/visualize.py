"""
Visualization functions for the housing price direction model.

Every function saves its output to outputs/ and optionally displays it
inline.  The outputs/ directory is created automatically if it does not
exist.  All plots are saved at 150 dpi so they are readable in a README.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from housing_model.config import OUTPUTS_DIR, BACKTEST_START, BACKTEST_STEP, PREDICTORS, TARGET

# Consistent visual style across all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":     120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"    Saved -> {path.name}")


def _date_axis(ax: plt.Axes) -> None:
    """Apply a clean year-only date format to the x-axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


# ---------------------------------------------------------------------------
# Plot 1: Prediction scatter
# ---------------------------------------------------------------------------

def plot_predictions(
    data: pd.DataFrame,
    results: dict[str, dict],
    show: bool = False,
) -> None:
    """
    Scatter plot of inflation-adjusted prices over time, colored green where
    the model predicted correctly and red where it was wrong.

    One subplot per model.
    """
    plot_data = data.iloc[BACKTEST_START:].copy()
    n_models = len(results)
    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(14, 4 * n_models),
        sharex=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        preds = res["preds"]
        n     = min(len(preds), len(plot_data))
        true  = data[TARGET].iloc[BACKTEST_START : BACKTEST_START + n].astype(int).values
        colors = np.where(preds[:n] == true, "#27ae60", "#e74c3c")

        ax.scatter(
            plot_data.index[:n],
            plot_data["adj_price"].iloc[:n],
            c=colors, s=6, alpha=0.75, linewidths=0,
        )
        ax.set_title(f"{name}  —  green = correct, red = wrong")
        ax.set_ylabel("Real Price (CPI-adj, $)")
        _date_axis(ax)

    fig.suptitle(
        "Walk-Forward Backtest: Predicted Price Direction vs Actual",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "predictions_scatter.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Rolling accuracy
# ---------------------------------------------------------------------------

def plot_rolling_accuracy(
    results: dict[str, dict],
    data: pd.DataFrame,
    show: bool = False,
) -> None:
    """
    Line chart of per-fold accuracy across the backtest timeline.

    A flat 50% baseline is drawn to show whether the model beats chance
    and to highlight regime-specific performance.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # One date label per fold window
    fold_dates = data.index[BACKTEST_START::BACKTEST_STEP]

    colors = ["#2980b9", "#e67e22", "#8e44ad", "#16a085"]
    for (name, res), color in zip(results.items(), colors):
        accs = res["window_accuracies"]
        n = min(len(accs), len(fold_dates))
        ax.plot(
            fold_dates[:n], accs[:n],
            marker="o", ms=4, linewidth=2,
            color=color, label=name,
        )

    ax.axhline(
        0.5, color="grey", linestyle="--",
        linewidth=1, label="Random baseline (50%)",
    )
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Backtest Window Start Date")
    ax.set_ylabel("Accuracy")
    ax.set_title("Rolling Accuracy per Backtest Window (52-week folds)")
    ax.legend(framealpha=0.9)
    _date_axis(ax)
    fig.tight_layout()
    _save(fig, "rolling_accuracy.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Simulated cumulative return
# ---------------------------------------------------------------------------

def plot_cumulative_return(
    data: pd.DataFrame,
    results: dict[str, dict],
    show: bool = False,
) -> None:
    """
    Simulated cumulative return of following each model's signal vs two baselines:
    - Always predict UP (long-only, buy-and-hold equivalent)
    - Random 50/50 coin-flip signal

    Each bar represents: "invest when signal = 1, hold cash when signal = 0."

    Note: Illustrative only.  Does not account for transaction costs,
    leverage, financing costs, or the illiquidity of housing markets.
    """
    plot_data = data.iloc[BACKTEST_START:].copy()
    weekly_returns = plot_data["adj_price"].pct_change().fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))

    # Baseline 1: always long
    always_up = (1 + weekly_returns).cumprod()
    ax.plot(
        plot_data.index, always_up,
        color="grey", linewidth=1.5, linestyle="--",
        label="Always Up (Long-Only)",
    )

    # Baseline 2: random signal
    rng = np.random.default_rng(42)
    random_sig   = pd.Series(rng.integers(0, 2, len(weekly_returns)), index=weekly_returns.index)
    random_cumret = (1 + weekly_returns * random_sig).cumprod()
    ax.plot(
        plot_data.index, random_cumret,
        color="grey", linewidth=1, linestyle=":",
        label="Random 50/50",
    )

    # Model signals
    model_colors = ["#2980b9", "#e67e22", "#8e44ad"]
    for (name, res), color in zip(results.items(), model_colors):
        preds  = res["preds"]
        n      = min(len(preds), len(weekly_returns))
        signal = pd.Series(preds[:n].astype(float), index=weekly_returns.index[:n])
        cumret = (1 + weekly_returns.iloc[:n] * signal).cumprod()
        ax.plot(cumret.index, cumret, color=color, linewidth=2, label=name)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base = 1.0)")
    ax.set_title(
        "Simulated Cumulative Return by Model Signal\n"
        "(Illustrative only — does not account for transaction costs or illiquidity)"
    )
    ax.legend(framealpha=0.9)
    _date_axis(ax)
    fig.tight_layout()
    _save(fig, "cumulative_return.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: SHAP summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    model,
    X_shap: pd.DataFrame,
    model_name: str = "model",
    show: bool = False,
) -> None:
    """
    SHAP beeswarm summary plot for the full backtested period.

    Uses the final fold's trained model applied to all rows from
    BACKTEST_START onward (n=648).  The final model was never trained
    on any of those rows in its own fold.

    SHAP shows both magnitude and direction: which features push the
    prediction toward "price up" (positive SHAP) or "price down" (negative).
    """
    try:
        import shap
    except ImportError:
        print("    shap not installed; skipping SHAP plot. Run: pip install shap")
        return

    print(f"    Computing SHAP values for {model_name}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_shap, check_additivity=False)

    # Normalise to a single 2-D array of shape (n_samples, n_features)
    # representing SHAP values for the positive class ("price up").
    #
    # RandomForest binary: returns list [neg_sv (n,f), pos_sv (n,f)]
    # XGBoost binary:      returns 2-D ndarray (n, f) directly
    # Interaction values:  returns 3-D ndarray (n, f, f) — collapse to 2-D
    if isinstance(sv, list):
        sv = np.array(sv[1])          # positive class
    if sv.ndim == 3:
        sv = sv.sum(axis=-1)          # sum interactions → main effects (n, f)

    # Let SHAP own the figure, then overwrite its auto-title cleanly
    shap.summary_plot(
        sv, X_shap,
        show=False,
        plot_type="dot",
        color_bar=True,
        max_display=24,
    )
    fig = plt.gcf()
    fig.axes[0].set_title(
        f"SHAP Feature Importance — {model_name}  (n={len(X_shap)} samples)",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    _save(fig, f"shap_{model_name.lower().replace(' ', '_')}.png")
    if show:
        plt.show()
    plt.close("all")


# ---------------------------------------------------------------------------
# Plot 5: Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    data: pd.DataFrame,
    predictors: list[str] = PREDICTORS,
    show: bool = False,
) -> None:
    """
    Lower-triangle correlation heatmap of all predictors.

    Shows multicollinearity at a glance so readers can understand which
    features carry redundant information.
    """
    subset = data.iloc[BACKTEST_START:][predictors].dropna()
    corr   = subset.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        annot=False,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Predictor Correlation Matrix (lower triangle)", pad=15)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")
    if show:
        plt.show()
    plt.close(fig)
