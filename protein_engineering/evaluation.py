from typing import Callable, Optional, Literal, Sequence

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchmetrics as tm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pytorch_lightning as pl

from sklearn.metrics import precision_score


def precision_recall_better_than_wt(mutations : pd.DataFrame, topk : int, wt_fitness : float = 0.0):
    """
    Fitness values are considered to be log-transformed. E.g. wildtype_fitness = 0.0
    """
    tops = mutations.nlargest(topk, 'model_confidence')
    actual = (tops['fitness'] >= wt_fitness).sum()
    total = (mutations['fitness'] >= wt_fitness).sum()

    precision = 1.0 * actual / topk
    recall = 1.0 * actual / total

    return (precision, recall)

def ndcg(mutations : pd.DataFrame, topk : int, wt_fitness : float = 0.0):
    """
    Fitness values are considered to be log-transformed by default. E.g. wildtype_fitness = 0.0
    """
    tops = mutations.nlargest(topk, 'model_confidence')
    gain = tops['fitness'].where(tops['fitness'] >= wt_fitness, 0.0)
    rank = np.arange(1, topk + 1)
    discounted_gain = gain / np.log2(rank + 1)
    dcg = discounted_gain.sum()

    actual = tops.sort_values(by='fitness')
    ideal = actual['fitness'].where(actual['fitness'] >= wt_fitness, 0.0)
    ideal_gain = ideal / np.log2(rank + 1)
    idcg = ideal_gain.sum()

    return dcg / idcg

def scatter_batch(
    model: pl.LightningModule,
    batch: dict[str, torch.Tensor],
    annotate_samples: bool = False,
    color_func: Optional[Callable[[Sequence[str | int]], Sequence[int | str]]] = None,
    **kwargs,
) -> plt.Figure:
    target = batch["y"]
    preds = model.predict_step(batch)
    if annotate_samples:
        annotate_samples = batch["id"]
    colors = list(map(color_func, batch["id"])) if color_func else None
    fig = scatter_plot(
        preds=preds, target=target, colors=colors, annotate_samples=annotate_samples, **kwargs
    )
    return fig


def scatter_plot(
    *,
    preds: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    colors: Optional[Sequence[int]] = None,
    annotate_metrics: bool = False,
    annotate_samples: Optional[Sequence[str]] = None,
    palette: str = "viridis",
    legend_title: str = None,
    **jointplot_kwargs,
) -> plt.Figure:
    lims = np.floor(target.min()), np.ceil(target.max())
    fig = sns.jointplot(
        x=target, y=preds, hue=colors, kind="scatter", palette=palette, **jointplot_kwargs
    )
    plt.plot(lims, lims, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    if annotate_samples:
        assert len(annotate_samples) == len(target)
        for i in range(len(target)):
            plt.annotate(
                annotate_samples[i],
                (target[i], preds[i]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=4,
            )
    if annotate_metrics:
        preds = torch.as_tensor(preds)
        target = torch.as_tensor(target)
        r2 = tm.functional.r2_score(preds, target)
        # rmse = tm.functional.mean_squared_error(preds, target, squared=False)
        # mae = tm.functional.mean_absolute_error(preds, target)
        pearson = tm.functional.pearson_corrcoef(preds, target)
        spearman = tm.functional.spearman_corrcoef(preds, target)
        plt.annotate(f"R2 = {r2:.2f}", (0.05, 0.95), xycoords="axes fraction", fontsize=8)
        # plt.annotate(f"RMSE = {rmse:.2f}", (0.05, 0.90), xycoords="axes fraction", fontsize=8)
        # plt.annotate(f"MAE = {mae:.2f}", (0.05, 0.85), xycoords="axes fraction", fontsize=8)
        plt.annotate(f"Pearson = {pearson:.2f}", (0.05, 0.80), xycoords="axes fraction", fontsize=8)
        plt.annotate(
            f"Spearman = {spearman:.2f}", (0.05, 0.75), xycoords="axes fraction", fontsize=8
        )

    if colors is not None:
        plt.legend(
            title=legend_title, loc="lower right", fontsize=8, title_fontsize=8, markerscale=0.5
        )

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()

    return fig.figure


### Custom calculation functions ###
def binned_reduce(
    x: torch.Tensor,
    values: torch.Tensor,
    bin_edges: torch.Tensor,
    reduce: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fill_value: Optional[float] = float("nan"),
    include_left_right: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce values in bins defined by bin_edges with a given reduction function.
    Similar to `scipy.stats.binned_statistic` but for tensors.

    Args:
        x: (torch.Tensor): Index according to which values are binned.
        values (torch.Tensor): Values to be reduced.
        bin_edges (torch.Tensor): Edges of the bins in which to reduce the values.
        reduce: (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Reduce function.
        fill_value (Optional[float], optional): Fill value for bins with no observations. Defaults to None.
        include_left_right (bool, optional): Whether to include bin form (-inf, bins[0]) and (bins[-1], inf). Defaults to False.

    Returns:
        torch.Tensor: Reduced values in bins.
                      The returned shape is (len(bin_edges) + 1) if include_left_right else (len(bin_edges) - 1)
        torch.Tensor: Number of observations in each bin.
    """
    _inf = torch.as_tensor([torch.inf])
    bin_edges_with_inf = torch.cat((-_inf, bin_edges, _inf))
    bin_values = torch.zeros(len(bin_edges) + 1)
    counts = torch.zeros(len(bin_edges) + 1, dtype=torch.long)

    for i, (low, high) in enumerate(zip(bin_edges_with_inf[:-1], bin_edges_with_inf[1:])):
        ids = torch.where((x > low) & (x <= high))[0]
        if len(ids) == 0:
            bin_values[i] = fill_value
            counts[i] = 0
        else:
            bin_values[i] = reduce(values[ids], x[ids])
            counts[i] = len(ids)

    if not include_left_right:
        bin_values = bin_values[1:-1]
        counts = counts[1:-1]
    return bin_values, counts


### Plotting functions ###
def plot_retrieval_precision_recall(
    precision: torch.Tensor, recall: torch.Tensor, k: torch.Tensor, ax: plt.Axes = None
) -> plt.Axes:
    """Plot retrieval precision and recall curves.
    For use with `torchmetrics.functional.retrieval_precision_recall_curve`.

    Args:
        precision (torch.Tensor): Precision values.
        recall (torch.Tensor): Recall values.
        k (torch.Tensor): Number of selected/retrieved variants.

    Returns:
        plt.Axes: Axes object.
    """
    max_k = k.max()
    if ax is None:
        ax = plt.gca()

    # Calculate precision & recall AUC
    precision_auc = tm.functional.auc(k / max_k, precision)
    recall_auc = tm.functional.auc(k / max_k, recall)

    # Plot
    ax.plot(k, precision, drawstyle="steps-post", label=f"Precision (AUC={precision_auc:.2f})")
    ax.plot(k, recall, drawstyle="steps-post", label=f"Recall (AUC={recall_auc:.2f})")

    # y-axis
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([str(i) + "%" for i in range(0, 110, 10)])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)

    # x-axis
    ax.set_xticks([1] + [10 * i for i in range(1, max_k + 1)])
    ax.set_xlabel("Number of selected variants (top-k)")
    ax.set_xlim(1, max_k)

    # Beautify
    ax.legend()
    ax.grid(alpha=0.1)
    sns.despine()
    ax.set_title("Retrieval precision & recall")

    return ax


def plot_precision_recall(
    precision: torch.Tensor, recall: torch.Tensor, ax: plt.Axes = None
) -> plt.Axes:
    """Plot precision and recall curves.
    For use with `torchmetrics.functional.precision_recall_curve`.

    Args:
        precision (torch.Tensor): Precision values.
        recall (torch.Tensor): Recall values.

    Returns:
        plt.Axes: Axes object.
    """
    if ax is None:
        ax = plt.gca()

    # Plot
    ax.plot(recall, precision, drawstyle="steps-post")

    # Calculate F1 scores and plot as levels
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.3)
        ax.annotate("$F_1$={0:0.1f}".format(f_score), xy=(0.9, y[45] - 0.04))

    # Calculate AUC
    auc = tm.functional.auc(recall, precision)
    # annotate with textbox
    ax.annotate(f"AUC={auc:.3f}", xy=(0.8, 0.95), bbox=dict(boxstyle="round", fc="w", alpha=0.1))

    # Beautify
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.1)
    sns.despine()
    ax.set_title("Precision-Recall curve")

    return ax


def plot_roc_curve(fpr: torch.Tensor, tpr: torch.Tensor, ax: plt.Axes = None) -> plt.Axes:
    """Plot ROC curve.
    For use with `torchmetrics.functional.roc_curve`.

    Args:
        fpr (torch.Tensor): False positive rate values.
        tpr (torch.Tensor): True positive rate values.

    Returns:
        plt.Axes: Axes object.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(fpr, tpr, drawstyle="steps-post")
    # Add diagonal line
    ax.plot([0, 1], [0, 1], color="gray", alpha=0.3)

    # Calculate AUC
    auc = tm.functional.auc(fpr, tpr)
    # annotate with textbox
    ax.annotate(f"AUROC={auc:.3f}", xy=(0.8, 0.05), bbox=dict(boxstyle="round", fc="w", alpha=0.1))

    ax.set_xlabel("False Positive Rate")
    ax.set_xlim(0, 1)

    ax.set_ylabel("True Positive Rate")
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.1)
    ax.set_title("ROC curve")
    sns.despine()

    return ax


def plot_threshold_analysis(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    thres_as_logit: bool = True,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot precision, recall (TPR) and FPR curves for different thresholds.

    Args:
        y_pred (torch.Tensor): Predicted logits/log-fitness values vs wildtype or
            probabilities.
        y_true (torch.Tensor): True (binary) labels.
        thres_as_logit (bool, optional): Whether the threshold values are given as
            logits/log-fitness values. Defaults to True.
        ax (plt.Axes, optional): Axes object. Defaults to None.

    Returns:
        plt.Axes: Axes object.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate precision, recall and thresholds
    if thres_as_logit:
        y_pred = y_pred.sigmoid()
    precision, recall, thres = tm.functional.classification.binary_precision_recall_curve(
        y_pred, y_true
    )
    fpr, _, thres2 = tm.functional.classification.binary_roc(y_pred, y_true)
    if thres_as_logit:
        thres = thres.logit()
        thres2 = thres2.logit()

    # Plot
    ax.plot(thres, precision[1:], label="Precision", drawstyle="steps-post")
    ax.plot(thres, recall[1:], label="Recall (True-positive rate)", drawstyle="steps-post")
    ax.plot(thres2, fpr, label="False-positive rate", drawstyle="steps-post")

    # Beautify
    ax.set_xlabel("Threshold")
    ax.set_xlim(thres.min(), thres.max())
    ax.set_ylabel("Rate")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([str(i) + "%" for i in range(0, 110, 10)])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.1)
    sns.despine()
    ax.set_title("Threshold analysis")

    return ax


def plot_binned_metric(
    bin_edges: torch.Tensor, metric: torch.Tensor, counts: torch.Tensor = None, ax: plt.Axes = None
) -> plt.Axes:
    """
    Plots a binned metric as a bar chart.

    Args:
        bin_edges (torch.Tensor): The edges of the bins. Should have shape `(n_bins + 1,)`.
        metric (torch.Tensor): The value of the metric for each bin. Should have shape `(n_bins,)`.
        counts (torch.Tensor, optional): The number of samples in each bin. If provided, should have shape `(n_bins,)`. Defaults to None.
        ax (plt.Axes, optional): The matplotlib Axes object to plot on. If not provided, uses the current Axes. Defaults to None.

    Returns:
        plt.Axes: The matplotlib Axes object containing the plot.
    """

    if ax is None:
        ax = plt.gca()

    ax.bar(x=bin_edges[:-1], height=metric, width=bin_edges.diff() * 0.95, align="edge")

    # Add counts to the bottom of each bar
    if counts is not None:
        bin_centers = bin_edges[:-1] + bin_edges.diff() / 2.0
        for i, count in enumerate(counts):
            ax.text(
                bin_centers[i],
                0.01,
                f"{metric[i]:,.2f}\n(n={count.item()})",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round", fc="w", alpha=0.8),
            )

    # Beautify
    sns.despine()
    return ax


### Full reporting functionality
def plot_evaluations(
    y_pred: torch.Tensor, y_true: torch.Tensor, wt_threshold: float = 0.0, n_bins: int = 5, **kwargs
) -> plt.Figure:
    """Plot all evaluation metrics in a grid

    Args:
        y_pred (torch.Tensor): Predicted logits/log-fitness values vs wildtype or
            probabilities.
        y_true (torch.Tensor): True (binary) labels.
        wt_threshold (float, optional): Threshold for wildtype. Defaults to 0.0,
            which means that all values below 0.0 are considered worse than wildtype.
        **kwargs: Additional keyword arguments passed to `plot_threshold_analysis`.

    Returns:
        plt.Figure: Figure object with all plots as subplots.
    """

    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)
    y_true_binary = (y_true > wt_threshold).long()

    # Create metrics
    precision, recall, _ = tm.functional.classification.binary_precision_recall_curve(
        y_pred.sigmoid(), y_true_binary
    )
    fpr, tpr, _ = tm.functional.classification.roc(y_pred.sigmoid(), y_true_binary)

    # Create report
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    # Make bottom 3 rows smaller

    plot_precision_recall(precision, recall, ax=axes[0, 0])
    plot_roc_curve(fpr, tpr, ax=axes[0, 1])
    plot_retrieval_precision_recall(
        *tm.functional.retrieval_precision_recall_curve(
            y_pred, y_true_binary, max_k=min(100, len(y_pred))
        ),
        ax=axes[1, 0],
    )
    plot_threshold_analysis(y_pred, y_true_binary, thres_as_logit=True, ax=axes[1, 1], **kwargs)

    # Plot worse than wildtype vs better than wildtype metrics
    # Get min and max values without nan
    y_true_min = torch.min(y_true[~torch.isnan(y_true)]).floor()
    y_true_max = torch.max(y_true[~torch.isnan(y_true)]).ceil()
    bin_edges = torch.Tensor([y_true_min, wt_threshold, y_true_max])
    bin_spearman, bin_counts = binned_reduce(
        y_true, y_pred, bin_edges, reduce=tm.functional.spearman_corrcoef
    )
    bin_pearson, _ = binned_reduce(y_true, y_pred, bin_edges, reduce=tm.functional.pearson_corrcoef)

    plot_binned_metric(torch.arange(-1, 2), metric=bin_spearman, counts=bin_counts, ax=axes[2, 0])
    axes[2, 0].set_title("Spearman correlation")
    axes[2, 0].set_xticks([-0.5, 0.5])
    axes[2, 0].set_xticklabels(["Worse than WT", "Better than WT"])

    plot_binned_metric(torch.arange(-1, 2), metric=bin_pearson, counts=bin_counts, ax=axes[2, 1])
    axes[2, 1].set_title("Pearson correlation")
    axes[2, 1].set_xticks([-0.5, 0.5])
    axes[2, 1].set_xticklabels(["Worse than WT", "Better than WT"])

    # Plot metrics for each bin
    bin_edges = torch.arange(
        y_true_min,
        y_true_max + 1,
        (y_true_max - y_true_min + 1) / 5,
    )
    bin_spearman, bin_counts = binned_reduce(
        y_true, y_pred, bin_edges, reduce=tm.functional.spearman_corrcoef
    )
    bin_pearson, _ = binned_reduce(y_true, y_pred, bin_edges, reduce=tm.functional.pearson_corrcoef)

    plot_binned_metric(bin_edges, metric=bin_spearman, counts=bin_counts, ax=axes[3, 0])
    axes[3, 0].set_title("Spearman correlation")
    axes[3, 1].set_xlabel("True fitness (binned)")

    plot_binned_metric(bin_edges, metric=bin_pearson, counts=bin_counts, ax=axes[3, 1])
    axes[3, 1].set_title("Pearson correlation")
    axes[3, 1].set_xlabel("True fitness (binned)")

    # Plot RMSE and MAE for each bin
    bin_rmse, _ = binned_reduce(
        y_true,
        y_pred,
        bin_edges,
        reduce=lambda x, y: tm.functional.mean_squared_error(x, y, squared=False),
    )
    bin_mae, _ = binned_reduce(y_true, y_pred, bin_edges, reduce=tm.functional.mean_absolute_error)

    plot_binned_metric(bin_edges, metric=bin_rmse, counts=bin_counts, ax=axes[4, 0])
    axes[4, 0].set_title("RMSE")
    axes[4, 0].set_xlabel("True fitness (binned)")

    plot_binned_metric(bin_edges, metric=bin_mae, counts=bin_counts, ax=axes[4, 1])
    axes[4, 1].set_title("MAE")
    axes[4, 1].set_xlabel("True fitness (binned)")

    fig.suptitle("Evaluation report")
    plt.tight_layout()

    return fig


def generate_model_report(
    save_path: str,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_pred_wt: float = None,
    y_true_wt: float = None,
    **scatter_kwargs,
) -> str:
    """Generate a pdf-report for a model.

    Example usage:
    ```python
    generate_model_report(
        "model_report.pdf",
        y_pred,
        y,
        y_pred[np.where(srired.data.is_wildtype)[0]],
        y[np.where(srired.data.is_wildtype)[0]],
        colors=srired.data.hamming_to_wildtype.astype(int),
        legend_title="Mutations"
    )
    ```

    Args:
        save_path (str): Path to save the report to.
        y_pred (torch.Tensor): Predicted fitness values.
        y_true (torch.Tensor): True fitness values.
        y_pred_wt (float, optional): Predicted fitness of the wildtype. Defaults to None.
        y_true_wt (float, optional): True fitness of the wildtype. Defaults to None.
        **scatter_kwargs: Keyword arguments passed to `scatter_plot`.

    Returns:
        str: Path to the saved report.
    """

    with PdfPages(save_path) as pdf:
        # Plot predictions
        fig = scatter_plot(
            preds=y_pred,
            target=y_true,
            annotate_metrics=True,
            alpha=0.8,
            # Select a categorical palette
            palette=cc.cm["rainbow"],
            s=10,
            height=5,
            legend="full",
            **scatter_kwargs,
        )
        # Annotate wildtype with a star
        if (y_true_wt is not None) and (y_pred_wt is not None):
            plt.scatter(
                x=torch.atleast_1d(torch.as_tensor(y_true_wt)),
                y=torch.atleast_1d(torch.as_tensor(y_pred_wt)),
                color="black",
                label="Wildtype",
                marker="*",
            )
        # Set ylim according to y_pred with a 5% margin
        _y_min, _y_max = y_pred.min(), y_pred.max()
        plt.ylim(_y_min - 0.05 * (_y_max - _y_min), _y_max + 0.05 * (_y_max - _y_min))
        # Move legend to the right
        plt.legend(bbox_to_anchor=(1.65, 1), borderaxespad=0.0)
        plt.suptitle(f"Evaluated data (n={len(y_pred):,})", y=1.05)
        # Shade area below y=0
        _y_min, _ = plt.gca().get_ylim()
        plt.axhspan(ymin=_y_min, ymax=0, alpha=0.05, color="grey")
        # Shade area to the left of x=0
        _x_min, _ = plt.gca().get_xlim()
        plt.axvspan(xmin=_x_min, xmax=0, alpha=0.05, color="grey")

        pdf.savefig(fig, bbox_inches="tight")
        pdf.savefig(plot_evaluations(y_pred, y_true, wt_threshold=0.0))

    return save_path