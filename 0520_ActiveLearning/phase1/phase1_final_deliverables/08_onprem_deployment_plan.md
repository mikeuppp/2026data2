# 08 — On-Prem Deployment Plan

## 1. Goal

Deploy the AI inference service inside the client’s controlled infrastructure to avoid network latency and protect proprietary image/research data.

## 2. Proposed Runtime Components

- FastAPI inference server.
- AdaptiveDETR/DINOv3 model wrapper.
- OCR preprocessing module.
- Postprocessor.
- Uncertainty scorer.
- Local model registry.
- Label Studio connector.
- Local logs/metrics.
- Docker container.

## 3. Privacy Rules

Recommended defaults:

- Do not store analyzed images after inference.
- Store only metadata unless explicitly configured otherwise.
- Store review images only when selected for Label Studio.
- Avoid logging raw image data.
- Redact sensitive OCR fields from logs if needed.
- Keep all model files and data inside client infrastructure.

## 4. Docker Structure

Suggested services:

```yaml
services:
  inference-api:
    image: bacterial-colony-ai:latest
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /models/current
      ENABLE_UNCERTAINTY: "true"
      STORE_IMAGES: "false"
      LABEL_STUDIO_URL: http://label-studio:8080
    volumes:
      - ./models:/models
      - ./logs:/logs
      - ./review_queue:/review_queue
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  label-studio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    volumes:
      - ./label_studio_data:/label-studio/data
```

## 5. Performance Benchmarking

The client has a very high target throughput. Because high-resolution full-image inference with a large DINOv3/AdaptiveDETR model can be computationally heavy, Phase 1 should define benchmarking rather than guarantee final throughput.

Benchmark metrics:

- p50 latency.
- p95 latency.
- p99 latency.
- GPU memory use.
- Max batch size.
- Images/second by image size.
- OCR overhead.
- Postprocessing overhead.
- Uncertainty scoring overhead.

## 6. Scaling Options

If one GPU is insufficient:

- Batch inference.
- Async request queue.
- Multiple inference workers.
- Multiple GPUs.
- Triton inference server if needed.
- TensorRT/ONNX optimization if compatible.
- Mixed precision inference.
- OCR routing to reduce unnecessary head computation.

## 7. Failure Handling

The service should handle:

- Invalid image format.
- Oversized image.
- No detected plate.
- OCR failure.
- Model load failure.
- GPU out-of-memory.
- Label Studio unavailable.
- Unsupported head/class.

## 8. Phase 1 Output

Phase 1 delivers this deployment design. Actual Docker implementation happens in Phase 3.
