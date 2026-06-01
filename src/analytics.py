from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt

PREDICTIONS_PATH = "../data/MRI_brain_tumor/parquet/predictions"

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

spark = (
    SparkSession.builder
    .appName("MRIAnalytics")
    .getOrCreate()
)

df = spark.read.parquet(
    PREDICTIONS_PATH
)

pdf = df.toPandas()

pdf["label"] = pdf["label"].map({
    "yes": 1,
    "no": 0
})

y_true = pdf["label"]
y_pred = pdf["prediction"]
y_prob = pdf["probability"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_prob
)

print("\n==============")
print("MODEL METRICS")
print("==============")

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "../analytics/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# class distribution
pdf["label"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Class Distribution"
)

plt.savefig(
    "../analytics/class_distribution.png",
    bbox_inches="tight"
)

plt.close()

# ROC curve
RocCurveDisplay.from_predictions(
    y_true,
    y_prob
)

plt.savefig(
    "../analytics/roc_curve.png",
    bbox_inches="tight"
)

plt.close()


df.createOrReplaceTempView(
    "predictions"
)

spark.sql(
"""
SELECT
label,
COUNT(*) AS total
FROM predictions
GROUP BY label
"""
).show()

spark.sql(
"""
SELECT
prediction,
COUNT(*) AS total
FROM predictions
GROUP BY prediction
"""
).show()


spark.sql(
"""
SELECT
label,
AVG(mean_intensity) AS avg_intensity,
AVG(std_intensity) AS avg_std
FROM predictions
GROUP BY label
"""
).show()

spark.stop()
