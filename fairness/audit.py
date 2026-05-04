import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import cohen_kappa_score


GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def run_fairness_audit(
    model,
    test_loader,
    test_df: pd.DataFrame,
    save_path: str
) -> pd.DataFrame:
    """
    Fairness audit across image quality subgroups.

    In the absence of demographic data, we use eye side and
    severity grouping as proxies to evaluate equity of model
    performance across subgroups — mirroring real-world
    clinical deployment equity requirements.
    """
    device = next(model.parameters()).device
    model.eval()

    # ── Collect predictions ───────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    results_df              = test_df.copy().reset_index(drop=True)
    results_df["predicted"] = all_preds
    results_df["correct"]   = (
        results_df["level"] == results_df["predicted"]
    ).astype(int)

    overall_acc = results_df["correct"].mean()

    # ── Subgroup 1: Per DR grade ──────────────────────────────────────────────
    print("=" * 55)
    print("Performance by DR Grade")
    print("=" * 55)
    grade_metrics = []
    for grade in range(5):
        subset = results_df[results_df["level"] == grade]
        if len(subset) == 0:
            continue
        acc = subset["correct"].mean()
        grade_metrics.append({
            "Grade": f"Grade {grade}\n{GRADE_NAMES[grade]}",
            "Accuracy": acc,
            "Count": len(subset)
        })
        print(f"  Grade {grade} ({GRADE_NAMES[grade]:15s}): "
              f"Acc={acc:.3f}  n={len(subset)}")

    # ── Subgroup 2: Left vs right eye ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Performance by Eye Side (Left vs Right)")
    print("=" * 55)
    results_df["eye_side"] = results_df["image"].apply(
        lambda x: "Left" if "left" in str(x).lower() else "Right"
    )
    side_metrics = []
    for side in ["Left", "Right"]:
        subset = results_df[results_df["eye_side"] == side]
        if len(subset) == 0:
            continue
        acc   = subset["correct"].mean()
        kappa = cohen_kappa_score(
            subset["level"], subset["predicted"], weights="quadratic"
        ) if len(subset["level"].unique()) > 1 else 0
        side_metrics.append({
            "Side": side, "Accuracy": acc,
            "Kappa": kappa, "Count": len(subset)
        })
        print(f"  {side:5s}: Acc={acc:.3f}  Kappa={kappa:.3f}  n={len(subset)}")

    # ── Subgroup 3: Severity group ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Performance by Severity Group")
    print("=" * 55)
    results_df["severity_group"] = results_df["level"].apply(
        lambda x: "No/Mild DR (0-1)" if x <= 1
        else "Moderate DR (2)" if x == 2
        else "Severe/Prolif. (3-4)"
    )
    for group in ["No/Mild DR (0-1)", "Moderate DR (2)", "Severe/Prolif. (3-4)"]:
        subset = results_df[results_df["severity_group"] == group]
        if len(subset) == 0:
            continue
        acc = subset["correct"].mean()
        print(f"  {group:22s}: Acc={acc:.3f}  n={len(subset)}")

    # ── Visualizations ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Fairness Audit — Model Performance Across Subgroups",
        fontsize=14
    )

    # Grade accuracy
    grade_df = pd.DataFrame(grade_metrics)
    axes[0].bar(grade_df["Grade"], grade_df["Accuracy"],
                color="steelblue", alpha=0.8)
    axes[0].axhline(y=overall_acc, color="red", linestyle="--",
                    label=f"Overall: {overall_acc:.3f}")
    axes[0].set_title("Accuracy by DR Grade")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Left vs right
    side_df = pd.DataFrame(side_metrics)
    x = np.arange(len(side_df))
    axes[1].bar(x, side_df["Accuracy"],
                color=["coral", "steelblue"], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{r['Side']}\n(n={r['Count']})" for _, r in side_df.iterrows()]
    )
    axes[1].axhline(y=overall_acc, color="red", linestyle="--",
                    label="Overall avg")
    axes[1].set_title("Accuracy: Left vs Right Eye")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # Severity group
    severity_acc = results_df.groupby("severity_group")["correct"].mean()
    axes[2].bar(
        range(len(severity_acc)), severity_acc.values,
        color=["green", "orange", "red"], alpha=0.8
    )
    axes[2].set_xticks(range(len(severity_acc)))
    axes[2].set_xticklabels(severity_acc.index, rotation=10, fontsize=9)
    axes[2].axhline(y=overall_acc, color="red", linestyle="--",
                    label="Overall avg")
    axes[2].set_title("Accuracy by Severity Group")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nFairness audit saved to {save_path}")

    return results_df