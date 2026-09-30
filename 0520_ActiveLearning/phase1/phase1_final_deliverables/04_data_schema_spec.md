# 04 — Data Schema Specification

## 1. Current JSON Response Structure

The uploaded inference JSON sample contains these top-level fields:

```json
[
  "boxes",
  "proba",
  "labels",
  "count",
  "backbone_label",
  "hough_circle",
  "backbone_label_string",
  "preferred_e_code",
  "preferred_l_code"
]
```

Observed sample summary:

| Field | Meaning |
|---|---|
| `boxes` | List of normalized bounding boxes. |
| `proba` | List of probability arrays, one array per head. |
| `labels` | List of label arrays, one array per head. |
| `count` | Count values per head. |
| `backbone_label` | Numeric global/domain label. |
| `backbone_label_string` | Human-readable selected head/domain. |
| `hough_circle` | Plate/circle metadata from preprocessing. |
| `preferred_e_code` | OCR-derived E-code. |
| `preferred_l_code` | OCR-derived L-code. |

In the sample:

- Number of boxes: `323`
- Number of heads: `8`
- Selected backbone label: `XSA`
- Preferred E-code: `E10/2026`
- Preferred L-code: `L446502`

## 2. Current Box Format

Boxes are normalized and appear to use:

```text
[x_center, y_center, width, height]
```

Example:

```json
[
  0.7876765131950378,
  0.18149660527706146,
  0.009505623951554298,
  0.008074549958109856
]
```

## 3. Current Multi-Head Output Pattern

Each head has one probability list and one label list of length equal to the number of predicted boxes.

Recommended fixed head order:

```text
0: ALL
1: EC
2: YM
3: XSA
4: ETB
5: LS
6: LM
7: BC
```

This head order should be confirmed by the client before implementation.

## 4. Proposed Normalized Response Schema

The production API can keep the existing fields for backward compatibility while adding a more explicit `predictions` list.

```json
{
  "image_id": "string",
  "model_version": "string",
  "backbone_label": 3,
  "backbone_label_string": "XSA",
  "preferred_e_code": "E10/2026",
  "preferred_l_code": "L446502",
  "hough_circle": [0.4935, 0.37625, 0.44, 0.36667],
  "boxes": [
    [0.7877, 0.1815, 0.0095, 0.0081]
  ],
  "proba": [],
  "labels": [],
  "count": [],
  "predictions": [
    {
      "box_index": 0,
      "box": [0.7877, 0.1815, 0.0095, 0.0081],
      "active_head_id": 3,
      "active_head_name": "XSA",
      "label": 0,
      "label_name": "XSA_class_0",
      "confidence": 0.9985,
      "uncertainty": 0.0015,
      "requires_review": false
    }
  ],
  "overall_uncertainty": 0.12,
  "review_required": false
}
```

## 5. Review/Active Learning Fields to Add

| Field | Type | Purpose |
|---|---|---|
| `uncertainty` | float | Per-box uncertainty score. |
| `requires_review` | boolean | Whether this prediction needs human verification. |
| `overall_uncertainty` | float | Whole-image active learning score. |
| `review_required` | boolean | Whether image should go to Label Studio. |
| `active_learning_reason` | string | Example: low confidence, high entropy, rare class, diversity selected. |
| `model_version` | string | Required for tracking feedback against model version. |

## 6. Feedback Schema

Corrected annotations from Label Studio should be converted into a consistent feedback format:

```json
{
  "image_id": "sample_001",
  "model_version": "adaptive_detr_dinov3_v1.2.0",
  "annotator_id": "analyst_01",
  "corrected_predictions": [
    {
      "box": [0.51, 0.44, 0.02, 0.02],
      "head_name": "XSA",
      "label": 1,
      "label_name": "XSA_positive",
      "correction_type": "label_changed"
    }
  ],
  "review_timestamp": "2026-06-03T10:00:00Z"
}
```

## 7. Phase 1 Schema Deliverable

The final Phase 1 schema deliverable should be reviewed by the client’s engineering team before Phase 2 implementation starts.
