import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for EfficientNet-B4.

    Highlights which regions of the fundus image
    drove the model's grade prediction.
    """

    def __init__(self, model):
        self.model       = model
        self.model.eval()
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last conv layer of EfficientNet backbone."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model.backbone.blocks[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, target_class=None):
        """Generate Grad-CAM++ heatmap for an image."""
        image_tensor = image_tensor.unsqueeze(0).to(
            next(self.model.parameters()).device
        )
        image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients   = self.gradients[0]
        activations = self.activations[0]

        grad_sq     = gradients ** 2
        grad_cube   = gradients ** 3
        denominator = 2 * grad_sq + activations.sum(dim=(1, 2), keepdim=True) * grad_cube
        denominator = torch.where(
            denominator != 0, denominator, torch.ones_like(denominator)
        )
        alpha   = grad_sq / denominator
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))

        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return (
            cam.cpu().numpy(),
            target_class,
            output.softmax(dim=1)[0].detach().cpu().numpy()
        )


def visualize_gradcam(
    model,
    test_loader,
    save_path: str,
    num_samples: int = 5
):
    """Visualize Grad-CAM++ heatmaps for sample test images."""
    grade_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    gradcam     = GradCAMPlusPlus(model)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))
    fig.suptitle(
        "Grad-CAM++ Explainability — Retinal Region Attribution",
        fontsize=14
    )

    for i in range(num_samples):
        img_tensor = images[i]
        true_label = labels[i].item()

        cam, pred_class, probs = gradcam.generate(img_tensor)

        img_display = img_tensor.clone()
        for c in range(3):
            img_display[c] = img_display[c] * std[c] + mean[c]
        img_np = img_display.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap     = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original\nTrue: {grade_names[true_label]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_resized, cmap="jet")
        axes[i, 1].set_title(
            f"Grad-CAM++ Heatmap\nPred: {grade_names[pred_class]}"
        )
        axes[i, 1].axis("off")

        correct = "✓" if pred_class == true_label else "✗"
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Overlay {correct}\nConf: {probs[pred_class]:.2%}"
        )
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM++ visualization saved to {save_path}")