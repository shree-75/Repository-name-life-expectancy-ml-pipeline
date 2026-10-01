# Life Expectancy Prediction

This project is a Life Expectancy version of the supplied Churn Prediction project. The same ML lifecycle is retained: data validation, leakage-safe preprocessing, baseline training, evaluation/error analysis, MLflow tracking, model registry, output validation, and reproducibility validation.

## Dataset
`data/raw/Life Expectancy Data.csv` contains 2,938 records and 22 columns.

### Prediction target
`Life expectancy ` (years) is the target, so this is a **regression** problem rather than classification. Rows where the target itself is missing are removed. Other missing feature values are handled by the preprocessing pipeline.

## Project structure
- `data/raw/` - original dataset
- `data/processed/` - transformed train/test arrays and metadata
- `models/` - saved preprocessing and model artifacts
- `src/` - validation, preprocessing, training, evaluation, MLflow, registry, and reproducibility code
- `pipelines/` - lab-style runners matching the Churn project
- `artifacts/` - reports and plots
- `outputs/` - prediction error analysis
- `notebooks/` - implementation notebook

## Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from Command Prompt or use `python -m pip install -r requirements.txt` without activating the environment.

## Main run
From the project root:
```powershell
python src/validate_data.py
python src/preprocess_pipeline.py
python src/validate_outputs.py
python src/train.py
python src/evaluate.py
python src/validate_reproducibility.py
```

Or run the complete lifecycle:
```powershell
python src/automate_lifecycle.py
```

## MLflow
After installing requirements and running preprocessing:
```powershell
python src/train_mlflow.py
mlflow ui
```
Then open the local MLflow URL shown in the terminal.

For model registry:
```powershell
python src/train_registry.py
```

## Important interpretation
The model predicts a continuous life-expectancy value in years. Report MAE, RMSE and R-squared; do not use churn classification metrics such as accuracy, precision, recall or ROC-AUC for this target.
