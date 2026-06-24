"""Paper-ready plotting helpers."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", context="paper")


def _save(fig, path=None):
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_pr_auc_robustness(results, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=results, x="split_year", y="pr_auc", hue="model", marker="o", ax=ax)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Temporal split year")
    return _save(fig, path)


def plot_roc_auc_robustness(results, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=results, x="split_year", y="roc_auc", hue="model", marker="o", ax=ax)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Temporal split year")
    return _save(fig, path)


def plot_reliability_curve(y_true, y_prob, path=None, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.plot(prob_pred, prob_true, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    return _save(fig, path)


def plot_calibration_bins(calibration_table, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    calibration_table.plot(kind="bar", x="bin", y=["predicted", "observed"], ax=ax)
    ax.set_ylabel("Probability")
    return _save(fig, path)


def plot_permutation_importance(importance, path=None, top_n=20):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = importance.head(top_n)
    sns.barplot(data=data, y="feature", x="importance_mean", ax=ax, color="#4C78A8")
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("")
    return _save(fig, path)


def plot_temporal_split_performance(results, metric="pr_auc", path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(data=results, x="split_year", y=metric, hue="model", ax=ax)
    ax.set_xlabel("Temporal split year")
    ax.set_ylabel(metric.upper())
    return _save(fig, path)


def plot_explainability_stability(stability, path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(stability.pivot(index="left", columns="right", values="spearman_rank_correlation"), annot=True, vmin=-1, vmax=1, ax=ax)
    return _save(fig, path)


def plot_confidence_intervals(ci_table, metric="estimate", path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ci_table["model"], ci_table[metric], yerr=[ci_table[metric] - ci_table["ci_lower"], ci_table["ci_upper"] - ci_table[metric]], fmt="o")
    ax.set_ylabel(metric)
    return _save(fig, path)


def plot_drift_heatmap(drift, path=None):
    matrix = drift.pivot(index="feature", columns="split_year", values="psi")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(matrix))))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Temporal split year")
    ax.set_ylabel("")
    return _save(fig, path)
