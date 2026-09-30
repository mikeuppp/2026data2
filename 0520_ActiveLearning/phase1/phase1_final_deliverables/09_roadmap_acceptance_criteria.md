# 09 — Roadmap and Acceptance Criteria

## Phase 2 — Core System Implementation

### Goal

Build the core Python modules for inference wrapping, uncertainty scoring, hybrid sampling, calibration, and pseudo-labeling.

### Deliverables

- Python package structure.
- Model wrapper interface.
- Uncertainty scoring module.
- Hybrid sampling module.
- Calibration utilities.
- Pseudo-labeling module.
- Active learning orchestrator.
- Unit tests.
- Example integration script.

### Acceptance Criteria

- Can accept outputs equivalent to the current JSON format.
- Can calculate per-box and per-image uncertainty.
- Can rank a batch of images by review priority.
- Can output a review queue file.
- Does not require exposing the client’s full proprietary code.

## Phase 3 — Inference Server and HITL Integration

### Goal

Package the system into an API server and connect it to Label Studio.

### Deliverables

- FastAPI server.
- Docker container.
- `/predict` endpoint.
- `/batch_predict` endpoint.
- `/uncertain_batch` endpoint.
- `/feedback` endpoint.
- Label Studio task export/import scripts.
- Basic model version handling.
- Retraining trigger script.

### Acceptance Criteria

- Client can send a Petri dish image and receive predictions.
- Server returns boxes, labels, counts, confidence, and uncertainty.
- Uncertain samples can be exported to Label Studio.
- Corrected labels can be imported back.
- The system runs on-prem or in the client’s chosen test environment.

## Phase 4 — Testing, Deployment, and Documentation

### Goal

Validate and hand off the system.

### Deliverables

- Test report.
- Latency report.
- Active learning simulation report.
- Deployment guide.
- API documentation.
- Label Studio user guide.
- Retraining operation guide.
- 2-hour training session.

### Acceptance Criteria

- System works end-to-end on representative images.
- Client team can run inference.
- Client team can review uncertain samples.
- Client team can export corrected labels.
- Documentation is sufficient for internal operation.
- Known risks and next optimization steps are documented.

## Recommended Milestone Payment Logic

- Phase 1 approval: architecture/specification accepted.
- Phase 2 approval: core active learning module works on sample outputs.
- Phase 3 approval: API server and Label Studio workflow run.
- Phase 4 approval: deployment/testing/docs accepted.
