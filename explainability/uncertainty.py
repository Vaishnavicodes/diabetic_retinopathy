import numpy as np
import torch
import matplotlib.pyplot as plt


def enable_dropout(model: torch.nn.Module):
    """Enable dropout layers during inference for MC sampling."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model,
    image_tensor: torch.Tensor,
    n_samples: int = 30,
    device: str = "cuda"
):
    """
    Monte Carlo Dropout inference.

    Runs n_samples forward passes with dropout enabled.
    Variance across passes = model uncertainty.

    High variance = model is unsure → flag for human review
    Low variance  = model is confident → safe to act on

    Returns:
        mean_probs:  average probability per grade
        std_probs:   uncertainty per grade
        pred_class:  predicted grade
        uncertainty: overall uncertainty score (max std)
    """
    model.eval()
    enable_dropout(model)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    predictions  = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(image_tensor)
            probs  = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())

    predictions  = np.array(predictions).squeeze(1)  # (n_samples, num_classes)
    mean_probs   = predictions.mean(axis=0)
    std_probs    = predictions.std(axis=0)
    pred_class   = mean_probs.argmax()
    uncertainty  = std_probs.max()

    return mean_probs, std_probs, pred_class, uncertainty


def visualize_uncertainty(
    model,
    test_loader,
    save_path: str,
    num_samples: int = 6,
    uncertainty_threshold: float = 0.15
):
    """
    Visualize MC Dropout uncertainty for test images.

    Flags high-uncertainty predictions for human review —
    a critical feature for clinical deployment.
    """
    grade_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    device      = next(model.parameters()).device

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 3.5))
    fig.suptitle(
        "Monte Carlo Dropout — Prediction Uncertainty Quantification",
        fontsize=14
    )

    for i in range(num_samples):
        img_tensor = images[i]
        true_label = labels[i].item()

        mean_probs, std_probs, pred_class, uncertainty = mc_dropout_predict(
            model, img_tensor, n_samples=30, device=device
        )

        # Unnormalize image for display
        img_display = img_tensor.clone()
        for c in range(3):
            img_display[c] = img_display[c] * std[c] + mean[c]
        img_np = np.clip(img_display.permute(1, 2, 0).numpy(), 0, 1)

        flag    = "⚠ HIGH UNCERTAINTY" if uncertainty > uncertainty_threshold else "✓ Confident"
        correct = "✓" if pred_class == true_label else "✗"

        # Image panel
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"True: {grade_names[true_label]} | "
            f"Pred: {grade_names[pred_class]} {correct}\n{flag}"
        )
        axes[i, 0].axis("off")

        # Uncertainty bar chart
        x      = np.arange(len(grade_names))
        colors = ["green" if j == pred_class else "steelblue" for j in range(5)]
        axes[i, 1].bar(
            x, mean_probs, yerr=std_probs,
            color=colors, capsize=5, alpha=0.8
        )
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(grade_names, rotation=15, fontsize=9)
        axes[i, 1].set_ylabel("Probability")
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title(
            f"Grade Probabilities ± Uncertainty\n"
            f"Max uncertainty: {uncertainty:.3f}"
        )
        axes[i, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Uncertainty plots saved to {save_path}")