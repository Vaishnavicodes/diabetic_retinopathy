import torch
import torch.nn as nn


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for DR grading.

    Why not standard CrossEntropy?
    CrossEntropy treats all misclassifications equally.
    Predicting Grade 0 when truth is Grade 4 is penalized
    the same as predicting Grade 3 — clinically wrong.

    Ordinal loss penalizes predictions proportionally to
    how far they are from the true grade. Misclassifying
    Grade 4 as Grade 0 gets a much higher penalty than
    misclassifying it as Grade 3.
    """

    def __init__(self, num_classes: int = 5, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)

        target_expanded = targets.unsqueeze(1).expand(batch_size, self.num_classes)
        class_indices   = torch.arange(self.num_classes).unsqueeze(0).expand(batch_size, -1).to(logits.device)

        distances = (class_indices - target_expanded).abs().float()
        weights   = 1.0 + distances

        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        loss      = -(smooth_labels * log_probs * weights).sum(dim=1).mean()
        return loss