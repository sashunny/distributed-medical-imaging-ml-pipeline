from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

import pandas as pd
import joblib


FEATURES_PATH = "../data/MRI_brain_tumor/parquet/mri_features"
MODEL_PATH = "../models/random_forest.pkl"
OUTPUT_PATH = "../data/MRI_brain_tumor/parquet/predictions"


model = joblib.load(MODEL_PATH)

spark = (
    SparkSession.builder
    .appName("BatchInference")
    .getOrCreate()
)

df = spark.read.parquet(FEATURES_PATH)

def predict_fn(width,
               height,
               mean_intensity,
               std_intensity):

    X = pd.DataFrame(
        [[
            width,
            height,
            mean_intensity,
            std_intensity
        ]],
        columns=[
            "width",
            "height",
            "mean_intensity",
            "std_intensity"
        ]
    )

    return int(model.predict(X)[0])

predict_udf = udf(
    predict_fn,
    IntegerType()
)

pred_df = df.withColumn(
    "prediction",
    predict_udf(
        "width",
        "height",
        "mean_intensity",
        "std_intensity"
    )
)

pred_df.show(10)

(
    pred_df.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

spark.stop()
