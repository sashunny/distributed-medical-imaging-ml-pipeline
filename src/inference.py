from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import *

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
    # find prediction
    pred = int(model.predict(X)[0])

    # find probablity
    prob = float(
        model.predict_proba(X)[0][1]
    )

    return (pred,prob)


prediction_schema = StructType([
    StructField(
        "prediction",
        IntegerType(),
        True
    ),
    StructField(
        "probability",
        DoubleType(),
        True
    )
])

predict_udf = udf(
    predict_fn,
    prediction_schema
)

pred_df = df.withColumn(
    "pred",
    predict_udf(
        "width",
        "height",
        "mean_intensity",
        "std_intensity"
    )
)

pred_df.show(10)

pred_df = (
    pred_df
    .withColumn(
        "prediction",
        pred_df.pred.prediction
    )
    .withColumn(
        "probability",
        pred_df.pred.probability
    )
    .drop("pred")
)

(
    pred_df.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

spark.stop()
