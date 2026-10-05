# Homework 8: Deep Learning

Dataset: curly vs. straight hair images (800 train / 201 eval).

```bash
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
unzip -q data.zip
```

## Questions

**Q1. Loss function**
Which loss matches the model (single logit output)?
- `nn.MSELoss()` / `nn.BCEWithLogitsLoss()` / `nn.CrossEntropyLoss()` / `nn.CosineEmbeddingLoss()`

**Q2. Parameter count**
What is the total number of trainable parameters (`sum(p.numel() for p in model.parameters())`)?
- `896` / `11214912` / `15896912` / `20073473`

**Q3. Baseline train accuracy**
Train 10 epochs (no augmentation). What's the median of `train_accuracy` (rounded to 2 decimals)?

**Q4. Baseline train loss std**
What's the population std of `train_loss` (rounded to 3 decimals)?

**Q5. Augmented evaluation loss**
Train 10 more epochs with augmentation. What's the mean of `evaluation_loss` (rounded to 3 decimals)?

**Q6. Augmented evaluation accuracy**
What's the mean of the last 5 values of `evaluation_accuracy` (rounded to 2 decimals)?

## Run

```bash
uv run 08-deep-learning/homework_08.py
```

## Submit

https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw08
