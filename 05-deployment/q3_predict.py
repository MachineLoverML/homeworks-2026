import pickle
from pathlib import Path
from model import normalize_record

with Path("pipeline.bin").open("rb") as f:
    pipeline = pickle.load(f)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

record = normalize_record(lead)
prob = float(pipeline.predict_proba([record])[0, 1])
print(f"conversion_probability: {prob:.3f}")
