from pyspark.sql import SparkSession
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


FEATURES_PATH = "../data/MRI_brain_tumor/parquet/mri_features"
MODEL_PATH = "../models/random_forest.pkl"

spark = (
    SparkSession.builder
    .appName("MRITraining")
    .getOrCreate()
)

df = spark.read.parquet(FEATURES_PATH)

pdf = df.select(
    "width",
    "height",
    "mean_intensity",
    "std_intensity",
    "label"
).toPandas()

pdf["label"] = pdf["label"].map({
    "yes": 1,
    "no": 0
})

X = pdf[
    [
        "width",
        "height",
        "mean_intensity",
        "std_intensity"
    ]
]

y = pdf["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(classification_report(y_test, preds))

joblib.dump(clf, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")

spark.stop()
