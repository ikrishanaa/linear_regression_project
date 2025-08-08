# Linear Regression Project

**Objective:** Implement and evaluate simple & multiple linear regression using scikit-learn.

## Contents
- `src/linear_regression.py` : Main script. Loads dataset (default: sklearn California housing), trains simple and multiple linear regression models, evaluates MAE, MSE, R², and saves a regression plot.
- `requirements.txt` : Python dependencies.
- `outputs/` : Generated metrics and plot when the script is run.

## How to run

1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows (PowerShell)
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script (default uses sklearn's California housing dataset):
   ```bash
   python src/linear_regression.py
   ```

   To run using your own CSV file (must contain a numeric target column named `target`):
   ```bash
   python src/linear_regression.py --csv /path/to/your/data.csv --target target
   ```

4. Outputs:
   - `outputs/metrics.json` : MAE, MSE, R2 for both simple and multiple regression.
   - `outputs/regression_plot.png` : Scatter + regression line (for a chosen single feature).
   - `outputs/coefficients.json` : Coefficients of multiple regression.

## Notes
- The script demonstrates both simple (single feature) and multiple linear regression.
- If you provide a CSV, the script will attempt to use the specified `--feature` (for plotting) and `--target`.