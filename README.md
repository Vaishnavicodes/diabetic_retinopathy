# Diabetic Retinopathy Grading
### EfficientNet-B4 · Ordinal Regression · Grad-CAM++ · MC Dropout · Fairness Audit

A clinically-realistic deep learning pipeline for automated diabetic 
retinopathy severity grading from fundus photographs. Built with a focus 
on clinical deployment considerations: ordinal loss, uncertainty 
quantification, explainability, and fairness auditing.

---

## Results

| Metric | Value |
|---|---|
| Validation Quadratic Kappa | **0.7625** |
| Test Quadratic Kappa | **0.6678** |
| Test Accuracy | **45%** (5-class, random = 20%) |
| Best performing grade | Severe (F1 = 0.54) |
| Training data | 700 images (balanced, 5 grades) |

---

## What Makes This Different

Most DR grading projects train EfficientNet and report AUC. This project goes further:

**Clinical realism:**
- **Ordinal regression loss** — penalizes predictions proportionally to grade 
  distance. Misclassifying Grade 4 as Grade 0 costs far more than Grade 3. 
  Standard cross-entropy ignores this.
- **Label smoothing** — handles inter-grader disagreement in the Kaggle DR 
  dataset, a known source of label noise in clinical imaging.
- **Monte Carlo Dropout** — 30 inference passes produce uncertainty estimates 
  per prediction. High-uncertainty cases are flagged for human review — a 
  critical feature for clinical deployment.

**Explainability:**
- **Grad-CAM++** — highlights which retinal regions drove each prediction. 
  The model attends to clinically relevant structures: hemorrhages, exudates, 
  and the optic disc.

**Fairness:**
- **Subgroup audit** — performance evaluated across DR grades, eye side 
  (left vs right), and severity groups. Found a 10-point kappa gap between 
  left (0.72) and right (0.61) eye performance — a deployment equity finding 
  that warrants investigation.

---

## Architecture