# 05 — Active Learning and Uncertainty Design

## 1. Goal

Reduce manual annotation effort by selecting only the most valuable images/colonies for human review.

The system should not ask analysts to review every image. It should send only uncertain, novel, rare, or strategically valuable samples to Label Studio.

## 2. Recommended Strategy

Use hybrid active learning:

```text
Active learning score = uncertainty score + diversity score + rarity/priority score
```

This avoids a common failure mode where the model sends many similar uncertain images for review.

## 3. Per-Box Uncertainty

For every detected colony/box, compute one or more of:

### Least Confidence

```text
uncertainty = 1 - max_class_probability
```

### Margin Uncertainty

```text
uncertainty = top_1_probability - top_2_probability
```

Lower margin means higher uncertainty.

### Entropy

```text
entropy = -sum(p_i * log(p_i))
```

Higher entropy means the model is more confused.

## 4. Per-Image Uncertainty

A Petri dish image can contain hundreds or thousands of colonies. Whole-image uncertainty can be computed using:

```text
image_uncertainty = max(per_box_uncertainty)
```

or:

```text
image_uncertainty = mean(top_K_uncertain_boxes)
```

Recommended default:

```text
image_uncertainty = 0.5 * mean(top_20_box_uncertainties)
                  + 0.3 * max_box_uncertainty
                  + 0.2 * rare_class_priority
```

## 5. Diversity Sampling

To avoid selecting duplicate-looking images, use feature-space diversity:

1. Extract embeddings from the backbone or decoder features.
2. Cluster or compare candidate images.
3. Select a diverse subset from the uncertain pool.

Possible methods:

- CoreSet / k-center greedy.
- Embedding clustering.
- BADGE-style gradient embedding if practical.
- Simple cosine-distance filtering as a Phase 2 baseline.

## 6. Pseudo-Labeling Strategy

High-confidence predictions can be used as pseudo-labels only when strict safety rules are met.

Recommended rules:

- Confidence above a high threshold, e.g. `> 0.95`.
- Prediction passes calibrated confidence threshold.
- Prediction is stable across augmentations if TTA is used.
- Class is not known to be problematic or rare.
- Pseudo-labels are stored separately from human labels.
- Validation must prove pseudo-labels improve performance before promotion.

## 7. Calibration

Raw confidence is not always reliable. Use calibration to improve uncertainty quality:

- Temperature scaling.
- Expected Calibration Error (ECE).
- Reliability diagrams.
- Per-head threshold tuning.

## 8. Human Review Routing

A prediction/image should be sent to human review if:

- Confidence is below threshold.
- Entropy is above threshold.
- Margin is below threshold.
- The image belongs to a rare or underperforming class.
- The model detects unusual density or distribution.
- OCR result conflicts with model head prediction.
- The sample is selected by diversity sampling.

## 9. Why No Heavy Ensemble by Default

The client has stated that latency and RAM are important, and that a traditional 3–5 model ensemble is not preferred.

Therefore the recommended Phase 1 design uses:

- Single-model calibrated uncertainty.
- Feature diversity.
- OCR-guided routing.
- Optional lightweight expert routing.
- Optional test-time augmentation only for low-confidence samples, not for every request.

## 10. Output of Active Learning Module

The module should output:

```json
{
  "image_id": "sample_001",
  "overall_uncertainty": 0.78,
  "rank": 1,
  "review_required": true,
  "reasons": [
    "high_entropy",
    "low_margin",
    "rare_class_candidate"
  ],
  "selected_boxes": [0, 12, 58]
}
```
