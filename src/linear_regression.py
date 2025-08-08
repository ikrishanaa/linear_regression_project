"""
Linear Regression example script.

Usage:
    python src/linear_regression.py [--csv PATH] [--target TARGET] [--feature FEATURE]

By default uses sklearn.datasets.fetch_california_housing.
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_default():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    # target column name will be 'MedHouseVal' in this script
    df = df.rename(columns={data.target.name: "target"})
    return df

def load_csv(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")
    df = df.copy()
    df = df.rename(columns={target: "target"})
    return df

def train_and_eval(df, feature_for_plot=None, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    # drop NaNs
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.shape[1] < 2:
        raise ValueError("Need at least one feature column plus target.")
    # Separate X and y
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Multiple Linear Regression (all features)
    mr = LinearRegression()
    mr.fit(X_train, y_train)
    y_pred_mr = mr.predict(X_test)

    metrics = {
        "multiple": {
            "MAE": float(mean_absolute_error(y_test, y_pred_mr)),
            "MSE": float(mean_squared_error(y_test, y_pred_mr)),
            "R2": float(r2_score(y_test, y_pred_mr))
        }
    }

    # Coefficients
    coeffs = {"intercept": float(mr.intercept_), "coefficients": dict(zip(X.columns.tolist(), mr.coef_.tolist()))}
    with open(os.path.join(out_dir, "coefficients.json"), "w") as f:
        json.dump(coeffs, f, indent=2)

    # Simple Linear Regression: pick a single feature (first or provided)
    if feature_for_plot is None:
        feature_for_plot = X.columns[0]
    if feature_for_plot not in X.columns:
        feature_for_plot = X.columns[0]

    X_train_f = X_train[[feature_for_plot]]
    X_test_f = X_test[[feature_for_plot]]

    sr = LinearRegression()
    sr.fit(X_train_f, y_train)
    y_pred_sr = sr.predict(X_test_f)

    metrics["simple"] = {
        "feature": feature_for_plot,
        "MAE": float(mean_absolute_error(y_test, y_pred_sr)),
        "MSE": float(mean_squared_error(y_test, y_pred_sr)),
        "R2": float(r2_score(y_test, y_pred_sr)),
        "intercept": float(sr.intercept_),
        "slope": float(sr.coef_[0])
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot simple regression
    plt.figure(figsize=(8,6))
    plt.scatter(X_test_f, y_test, alpha=0.4, label="Actual")
    # regression line
    xs = np.linspace(X_test_f.min()[0], X_test_f.max()[0], 100).reshape(-1,1)
    ys = sr.predict(xs)
    plt.plot(xs, ys, linewidth=2, label="Regression line")
    plt.xlabel(feature_for_plot)
    plt.ylabel("target")
    plt.title(f"Simple Linear Regression — feature: {feature_for_plot}")
    plt.legend()
    out_plot = os.path.join(out_dir, "regression_plot.png")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    print("Saved outputs to", out_dir)
    return metrics, coeffs, out_plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file. If omitted, uses sklearn California housing dataset.")
    parser.add_argument("--target", type=str, default="target", help="Target column name if using CSV. Default 'target'.")
    parser.add_argument("--feature", type=str, default=None, help="Feature column name to use for simple regression plot.")
    args = parser.parse_args()

    if args.csv:
        df = load_csv(args.csv, args.target)
    else:
        df = load_default()

    metrics, coeffs, plot_path = train_and_eval(df, feature_for_plot=args.feature)
    print(json.dumps(metrics, indent=2))
    print("Coefficients saved to outputs/coefficients.json")
    print("Plot saved to", plot_path)

if __name__ == "__main__":
    main()