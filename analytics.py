import pandas as pd
from transformers import pipeline


def quick_analysis(csv_path: str):
    df = pd.read_csv(csv_path)

    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": categorical.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": numeric.describe().round(2).to_dict(),
        "top_categorical_values": {
            c: df[c].value_counts().head(5).to_dict()
            for c in categorical.columns
        }
    }

def sentiment_sample(csv_path: str, n: int = 200):
    df = pd.read_csv(csv_path).head(n).copy()
    df["text_summary"] = df["JobRole"].astype(str) + " | " + df["Department"].astype(str)

    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    out = clf(df["text_summary"].tolist())

    df["sentiment"] = [r["label"] for r in out]
    df["score"] = [float(r["score"]) for r in out]
    return df