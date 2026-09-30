# Current AdaptiveDETR Architecture

## 1. Global head/backbone classifier

The current AdaptiveDETR architecture includes a global classifier for the head/backbone. This classifier predicts which domain or head the complete image belongs to.

```python
self.backbone_classifier = MLP(
    backbone.num_channels,
    backbone.num_channels,
    backbone_num_classes,
    backbone_classifier_layers,
)
```

Current configuration:

```yaml
backbone_num_classes: 8
backbone_classifier_layers: 1
```

In the current configuration, this is basically a single linear layer:

```text
1536 -> 8
```

## 2. Domain-specific detection heads

There is a `ModuleList` containing one MLP per detection head:

```python
self.class_embeds = nn.ModuleList(
    MLP(hidden_dim, hidden_dim, num_classes + 1, detection_classifier_layers)
    for num_classes in self.num_det_classes
)
```

Current configuration:

```yaml
hidden_dim: 256
detection_classifier_layers: 1
num_classes: [1, 2, 2, 2, 1, 1, 3, 1]
```

Therefore, the model has 8 detection heads:

```text
ALL: 256 -> 2   # 1 class + background
EC:  256 -> 3   # 2 classes + background
YM:  256 -> 3
XSA: 256 -> 3
ETB: 256 -> 2
LS:  256 -> 2
LM:  256 -> 4   # 3 classes + background
BC:  256 -> 2
```

## 3. Shared box prediction head

In addition to the domain-specific classification heads, the model has a shared head for bounding box prediction:

```python
self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
```

This head is a 3-layer MLP that predicts bounding boxes in the following format:

```text
[x_center, y_center, width, height]
```

## 4. Current inference flow

The following is a simplified version of the current detector runtime.

```python
# 1. Preprocess
detection_input = DetectionInput(image=image)
batch = preprocessor(detection_input)

# 2. Global domain/head classification
classify_output = model(
    batch.original_image,
    backbone_class_ids=None,
    mode="classify",
)
backbone_logits = classify_output["backbone_logits"]

# 3. Detection inference with all heads
inference_output = model(
    batch.nested_tensor,
    backbone_class_ids=None,
    mode="inference",
)

pred_logits = inference_output["pred_logits"]  # list: one tensor per head
pred_boxes = inference_output["pred_boxes"]

# 4. Postprocess
output = postprocessor(
    ModelOutput(
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        backbone_logits=backbone_logits,
        hough_circle=batch.hough_circle,
    )
)
```

In words: the system first classifies the complete image to obtain the `backbone_label` or `head_id`. However, during `mode="inference"`, the model computes logits for all detection heads.

After that, the postprocessor uses the corresponding head to interpret labels, probabilities, counts, and final bounding boxes.