# 01 — Phase 1 Final Deliverables

## Phase 1 Name

Architecture Design & Specification

## Duration

1 week

## Phase 1 Objective

The objective of Phase 1 is to convert the client's existing AdaptiveDETR/DINOv3 architecture, inference JSON format, and active learning goals into a clear implementation blueprint.

This phase does not train a final model and does not deploy a production system. Instead, it defines exactly what will be built in Phases 2–4 so that implementation can proceed without ambiguity.

## Current System Facts Used in This Design

From the supplied architecture markdown:

- The current system has a global backbone/head classifier.
- The global classifier predicts one of 8 domains/heads.
- The current global classifier is effectively a `1536 -> 8` linear classifier.
- The model uses 8 domain-specific detection heads:
  - ALL
  - EC
  - YM
  - XSA
  - ETB
  - LS
  - LM
  - BC
- Bounding box prediction is shared across heads.
- The bounding box format is `[x_center, y_center, width, height]`.
- Current inference first performs global classification, then computes logits for all detection heads, then the postprocessor selects/interprets the relevant head.

From the supplied JSON sample:

- The current response includes keys: `boxes, proba, labels, count, backbone_label, hough_circle, backbone_label_string, preferred_e_code, preferred_l_code`.
- The sample contains `323` predicted boxes.
- The sample includes 8 probability arrays, 8 label arrays, and 8 count arrays.
- The selected/global label in the sample is `XSA`.
- The sample includes OCR-related metadata such as `preferred_e_code` and `preferred_l_code`.

## What I Will Deliver in Phase 1

### 1. Architecture Specification

A written system architecture describing:

- Existing model flow.
- Target inference server architecture.
- OCR + global routing logic.
- AdaptiveDETR/DINOv3 inference wrapper.
- Postprocessing structure.
- Active learning module placement.
- Label Studio integration.
- Model versioning and retraining path.
- On-premise deployment boundaries.

### 2. API Specification

A draft OpenAPI/Swagger API spec for:

- `/health`
- `/predict`
- `/batch_predict`
- `/uncertain_batch`
- `/feedback`
- `/model/version`

This provides the contract between the AI server and the client’s end-user system.

### 3. Data Schema Specification

A normalized schema for:

- Input image request.
- OCR/Petri dish metadata.
- Backbone/domain classification.
- Bounding boxes.
- Per-head predictions.
- Counts.
- Uncertainty values.
- Human-review flags.
- Feedback records.
- Retraining metadata.

### 4. Active Learning Strategy

A design for:

- Per-box confidence scoring.
- Per-image uncertainty scoring.
- Hybrid sampling using uncertainty + diversity.
- Human review threshold logic.
- Pseudo-labeling safety rules.
- Selection of images for annotation.
- Avoiding redundant annotation requests.

### 5. Label Studio Workflow

A planned on-prem Label Studio workflow:

- Export uncertain images/tasks.
- Pre-fill model predictions.
- Allow lab analysts to correct outputs.
- Import corrected annotations.
- Convert corrections into retraining-ready format.

### 6. Training and Retraining Strategy

A written plan for:

- Baseline evaluation.
- Model improvement.
- Backbone freezing/fine-tuning decision.
- Head-specific retraining.
- Validation metrics.
- Model promotion criteria.
- Rollback/versioning.

### 7. On-Prem Deployment Plan

A deployment blueprint covering:

- FastAPI server.
- Docker packaging.
- GPU runtime assumptions.
- No image retention policy.
- Logging without storing proprietary image data.
- Local model registry.
- Throughput benchmarking plan.

### 8. Implementation Roadmap

A refined plan for Phases 2–4, including:

- What will be built.
- What the client must provide.
- Acceptance criteria.
- Risks and open questions.

## What the Client Must Provide During Phase 1

The client does not need to share the full proprietary codebase during Phase 1. However, the following items are needed:

1. Confirmation of final head order and class mapping.
2. Current inference JSON specification or representative examples.
3. A few sample images and expected outputs.
4. Description of preprocessing and postprocessing steps.
5. Deployment hardware details.
6. Expected latency/throughput targets.
7. Label Studio hosting preference.
8. Any security restrictions.
9. Clarification of whether OCR routing should decide the active expert/head.
10. Whether historical unlabeled archives are available for SSL/pseudo-labeling.

## Phase 1 Exit Criteria

Phase 1 is considered complete when the client receives and approves:

- Architecture specification.
- API contract.
- Data schema specification.
- Active learning design.
- Label Studio workflow plan.
- Training/retraining strategy.
- Deployment plan.
- Phase 2–4 roadmap.
