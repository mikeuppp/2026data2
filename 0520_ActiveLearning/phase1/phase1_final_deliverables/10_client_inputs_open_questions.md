# 10 — Client Inputs and Open Questions

## Required Before Phase 2

1. Final head order:
   - Confirm: ALL, EC, YM, XSA, ETB, LS, LM, BC.

2. Final class names:
   - For each head, provide numeric label -> human class name mapping.

3. Current JSON response details:
   - Confirm whether the uploaded JSON is the exact production response or only a compact/sample version.

4. Image input requirements:
   - Supported formats.
   - Maximum file size.
   - Typical image resolution.
   - Whether images are RGB/BGR/grayscale.
   - Preprocessing steps.

5. OCR details:
   - OCR model output format.
   - Whether OCR is mandatory for routing.
   - How E-code and L-code should be used.

6. Deployment hardware:
   - GPU model(s).
   - CPU/RAM.
   - OS.
   - Docker/NVIDIA runtime availability.

7. Throughput/latency target:
   - Confirm realistic production target.
   - Define acceptable p95 latency.
   - Confirm whether batching is allowed.

8. Label Studio:
   - Server URL or planned hosting method.
   - User roles.
   - Annotation schema.
   - Export format preference.

9. Data privacy:
   - Can selected review images be stored locally?
   - How long can review images be retained?
   - Are logs allowed to store OCR strings?

10. Training:
   - Number of annotated images available.
   - Number of unlabeled historical images.
   - Validation/test split rules.
   - Whether cloud training is allowed for non-sensitive or anonymized data.

## Important Open Technical Questions

### 1. Should inference compute all heads or only the OCR/global-selected head?

Current architecture computes logits for all detection heads during inference. This is flexible but may be expensive at high throughput.

Possible decision:

- Keep all heads for maximum safety.
- Use OCR/global routing for speed.
- Use all heads only when OCR confidence is low.

### 2. How should uncertainty be calculated across multiple heads?

Possible options:

- Use only active head uncertainty.
- Compare active head confidence with other heads.
- Use OCR/head disagreement as uncertainty.
- Use per-head thresholds.

### 3. Should query count increase to 1,000?

Increasing queries may help with dense colonies, but it can increase memory, latency, and postprocessing cost. This should be benchmarked.

### 4. How will pseudo-labels be validated?

Pseudo-labeling can reduce manual annotation but can also amplify model errors. It should be introduced with conservative thresholds and validation.

### 5. What is the minimum acceptable model performance?

The client should define:

- Minimum F1 per head.
- Maximum acceptable false negatives.
- Count accuracy tolerance.
- Latency target.
- Calibration target.

## Phase 1 Recommendation

Before implementation begins, hold one technical review meeting to approve:

- API format.
- Head/class mapping.
- Active learning selection logic.
- Deployment assumptions.
- Phase 2 acceptance criteria.
