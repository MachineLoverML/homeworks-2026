# Homework 5: Deployment

Model: lead-scoring pipeline (`pipeline.bin`) trained on Course Lead Scoring 2026 dataset.

## Questions

**Q1. Environment check**
Run `uv --version` and record it. (Setup check, not graded.)

**Q2. Locked Scikit-Learn version**
Which Scikit-Learn version is in `pyproject.toml` / `uv.lock`?
- `1.6.1` / `1.7.2` / `1.8.0` / `2.0.0`

**Q3. Load the model**
Load `pipeline.bin` with `pickle`, run `predict_proba` on the provided lead. What's the conversion probability (rounded to 3 decimals)?

**Q4. Serve the model**
Start the API with `uv run uvicorn predict:app --host 0.0.0.0 --port 9696`, send the second lead via `POST /predict`. What's `conversion_probability` (rounded to 3 decimals)?

**Q5. Container base image**
Which Python base-image tag is in the Dockerfile?
- `python:3.9-slim-bullseye` / `python:3.11.15-slim-bookworm` / `python:3.12-slim-bookworm` / `python:3.13.10-slim-bookworm`

**Q6. Run the container**
Build and run the Docker container, repeat Q4's request. What's `conversion_probability` (rounded to 3 decimals)?

## Run

```bash
cd cohorts/2026/homework/05-deployment
uv sync --locked
uv run python smoke_test.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05
