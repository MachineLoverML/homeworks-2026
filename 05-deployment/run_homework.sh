#!/usr/bin/env bash
# Run from: machine-learning-zoomcamp/cohorts/2026/homework/05-deployment/

# Q1 — uv version (not graded)
uv --version

# Verify model artifact checksum
sha256sum pipeline.bin
# expected: 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1

# Setup — install locked dependencies
uv sync --locked

# Q3 — predict_proba for lead 1 (answer: 0.533)
uv run python q3_predict.py

# Q4 — serve the model and POST lead 2 (answer: 0.770)
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 &
UVICORN_PID=$!
sleep 2

curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'

kill $UVICORN_PID

# Q5 — base image tag is declared in Dockerfile (answer: python:3.11.15-slim-bookworm)
grep '^FROM' Dockerfile

# Q6 — build and run the container, same request as Q4 (answer: 0.770)
docker build -t zoomcamp-model:2026-hw5 .

docker run --rm -d -p 9696:9696 --name zoomcamp-hw5 zoomcamp-model:2026-hw5
sleep 3

curl -s http://localhost:9696/health

curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'

docker stop zoomcamp-hw5
