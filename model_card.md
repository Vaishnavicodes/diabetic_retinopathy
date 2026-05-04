# Model Card: Diabetic Retinopathy Grading

## Model Details

| Field | Details |
|---|---|
| Model name | DR-EfficientNet-B4 |
| Version | 1.0 |
| Model type | Image classification — ordinal regression |
| Architecture | EfficientNet-B4 + custom ordinal head |
| Framework | PyTorch 2.x + timm |
| License | MIT |
| Contact | your.email@example.com |

---

## Intended Use

### Primary intended use
Assistive tool for diabetic retinopathy severity grading from fundus 
photographs. Intended to support — not replace — clinical decision-making 
by trained ophthalmologists and optometrists.

### Primary intended users
- Ophthalmology researchers
- Clinical AI developers
- Medical imaging data scientists

### Out-of-scope uses
- Autonomous clinical diagnosis without physician oversight
- Deployment in resource-limited settings without validation on 
  local patient populations
- Use on non-fundus images or non-DR eye conditions
- Any use that bypasses qualified medical review

---

## Training Data

| Field | Details |
|---|---|
| Dataset | Kaggle Diabetic Retinopathy Detection (EyePACS) |
| Source | kaggle.com/c/diabetic-retinopathy-detection |
| Training samples | 700 (stratified sample, 140 per grade) |
| Validation samples | 150 (30 per grade) |
| Test samples | 150 (30 per grade) |
| Label source | Clinical graders (EyePACS protocol) |
| Known label noise | Inter-grader disagreement — addressed via label smoothing |

### Data preprocessing
- Black border cropping to isolate retinal tissue
- CLAHE enhancement for lesion visibility
- Resize to 512×512
- ImageNet normalization

### Training augmentation
- Random horizontal/vertical flip
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation)

---

## Evaluation Results

### Overall performance

| Metric | Validation | Test |
|---|---|---|
| Quadratic Weighted Kappa | 0.7625 | 0.6678 |
| Accuracy | ~50% | 45% |
| Macro F1 | — | 0.44 |

### Per-grade performance (test set)

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — No DR | 0.38 | 0.30 | 0.33 | 30 |
| 1 — Mild | 0.39 | 0.50 | 0.44 | 30 |
| 2 — Moderate | 0.47 | 0.30 | 0.37 | 30 |
| 3 — Severe | 0.46 | 0.63 | 0.54 | 30 |
| 4 — Proliferative | 0.54 | 0.50 | 0.52 | 30 |

### Fairness audit

| Subgroup | Accuracy | Kappa |
|---|---|---|
| Left eye | 0.457 | 0.722 |
| Right eye | 0.438 | 0.613 |
| No/Mild DR (grades 0-1) | 0.400 | — |
| Moderate DR (grade 2) | 0.300 | — |
| Severe/Proliferative (grades 3-4) | 0.567 | — |

**Key finding:** A 10-point kappa gap between left (0.722) and right 
(0.613) eye performance was identified. This may reflect systematic 
differences in fundus image acquisition protocols between eye sides 
and warrants investigation before clinical deployment.

---

## Model Architecture & Design Decisions

### Why EfficientNet-B4?
EfficientNet-B4 offers the best accuracy/efficiency tradeoff for 
medical image classification at 512×512 resolution. It outperforms 
ResNet-50 on DR grading tasks while being significantly smaller than 
EfficientNet-B7.

### Why ordinal regression loss?
DR severity grades are ordered (0 < 1 < 2 < 3 < 4). Standard 
cross-entropy treats all misclassifications equally — clinically 
inappropriate. Our ordinal loss penalizes predictions proportionally 
to grade distance, encoding clinical severity ordering into the 
training objective.

### Why Monte Carlo Dropout?
Softmax confidence scores are overconfident and poorly calibrated. 
MC Dropout samples the posterior distribution of model weights across 
30 inference passes, producing calibrated uncertainty estimates. 
Predictions with uncertainty > 0.15 are flagged for human review.

### Why CLAHE?
Fundus image contrast varies significantly across imaging devices and 
clinical settings. CLAHE normalizes local contrast to improve lesion 
visibility regardless of camera quality — critical for generalization 
across diverse clinic environments.

---

## Limitations

### Known limitations
- **Small training set:** Trained on 700 images. Full dataset (35k) 
  training expected to push kappa above 0.85.
- **Left/right eye gap:** 10-point kappa difference between eye sides 
  requires investigation.
- **Uncertainty calibration:** The 0.15 uncertainty threshold was 
  set heuristically. Proper calibration requires a dedicated 
  calibration dataset.
- **No demographic data:** Fairness audit used eye side and severity 
  as proxies. Demographic subgroup analysis (age, sex, ethnicity) 
  was not possible with available data.
- **Single dataset:** Trained and evaluated on EyePACS data only. 
  Performance on other fundus camera types is unknown.

### What the model cannot do
- Detect other eye conditions (glaucoma, AMD, cataracts)
- Grade image quality or flag ungradable images
- Provide a clinical diagnosis
- Replace ophthalmologist review

---

## Ethical Considerations

### Intended safeguards
- Model outputs include uncertainty scores to flag borderline cases
- Grad-CAM++ heatmaps enable clinician verification of model attention
- Fairness audit conducted to identify demographic performance gaps
- Model card published to ensure transparent reporting

### Risks
- **Overreliance:** Clinicians may defer to model predictions without 
  adequate review. Mitigation: uncertainty flagging and clear 
  "assistive tool" framing.
- **Distribution shift:** Model trained on EyePACS data may underperform 
  on fundus images from different cameras or populations.
- **Label noise propagation:** Inter-grader disagreement in training 
  labels may bias model toward majority grader opinion.

### Recommendations before clinical deployment
1. Validate on local patient population data
2. Conduct prospective clinical trial
3. Calibrate uncertainty threshold on validation cohort
4. Investigate left/right eye performance gap
5. Obtain regulatory clearance (FDA 510(k) or equivalent)

---

## Citation

```bibtex
@misc{dr-grading-2026,
  title   = {Diabetic Retinopathy Grading with Clinical Realism},
  author  = {Your Name},
  year    = {2026},
  url     = {https://github.com/yourusername/diabetic-retinopathy-grading}
}
```

---

## References

1. Gulshan et al. (2016). Development and Validation of a Deep Learning 
   Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus 
   Photographs. JAMA, 316(22), 2402–2410.

2. Selvaraju et al. (2020). Grad-CAM++: Improved Visual Explanations 
   for Deep Convolutional Networks. WACV 2020.

3. Tan & Le (2019). EfficientNet: Rethinking Model Scaling for 
   Convolutional Neural Networks. ICML 2019.

4. Gal & Ghahramani (2016). Dropout as a Bayesian Approximation: 
   Representing Model Uncertainty in Deep Learning. ICML 2016.