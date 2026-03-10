# Causality-technical-test (via wiremind paper)
This project implements demand forecasting models for train ticket pricing optimization using causal machine learning techniques.

## Files
- `EDA.ipynb` - Exploratory data analysis and dataset understanding
- `model-training-corrected.ipynb` - Data analysis, feature engineering, and model training (standard + causal)
- `model-evaluation.ipynb` - Model validation with causal sanity checks and revenue analysis

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn jupyter

# Run notebooks in order
jupyter notebook model-training-corrected.ipynb  # Train models
jupyter notebook model-evaluation.ipynb          # Evaluate and compare
```

## Key Results
- **Standard LightGBM**: 37.8% wrong price elasticity ❌ (confounded)
- **Causal Model**: 0.0% wrong elasticity ✓ (safe for pricing optimization)

## Approach
Implemented orthogonal learning (Crasson et al. 2024) to fix confounding bias and enable reliable revenue optimization.
