# Homework 9: Serverless Deep Learning

Model: reference ONNX hair classifier (`hair_classifier_v1.onnx`).

```bash
curl -fL -o hair_classifier_v1.onnx.data "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data"
curl -fL -o hair_classifier_v1.onnx "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx"
curl -fL -o sample.jpeg "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
```

## Questions

**Q1. ONNX output node name**
What is the output node name from `session.get_outputs()`?
- `output` / `sigmoid` / `softmax` / `prediction`

**Q2. Target image size**
What size does `prepare_image` resize to?
- `64x64` / `128x128` / `200x200` / `256x256`

**Q3. Normalized input value**
Run `prepare_image` on `sample.jpeg`. What's the first R-channel value (`tensor[0,0,0,0]`, rounded to 3 decimals)?

**Q4. ONNX inference output**
Run inference on `sample.jpeg`. What's the output probability (rounded to 3 decimals)?

**Q5. Lambda base image**
Which runtime base image is in the Dockerfile?
- `public.ecr.aws/lambda/python:3.9` / `:3.11` / `:3.13` / `python:3.13-slim-bookworm`

**Q6. Container inference**
Build and run the Lambda container, invoke it with `sample.jpeg`. What's `straight_probability` (rounded to 3 decimals)?

## Run

```bash
uv run 09-serverless/homework_09.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw09
