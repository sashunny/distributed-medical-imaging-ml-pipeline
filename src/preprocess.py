from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import cv2
import numpy as np

INPUT_DIR =  "../data/MRI_brain_tumor/parquet/mri_metadata"
OUTPUT_DIR = "../data/MRI_brain_tumor/parquet/mri_features"

spark = (
    SparkSession.builder.appName("MRIPreprocessing").getOrCreate()
)

df = spark.read.parquet(INPUT_DIR)
print("Loaded rows:", df.count())


# FEATURE EXTRACTION

schema = StructType([
    StructField("width", IntegerType(), True),
    StructField("height", IntegerType(), True),
    StructField("mean_intensity", DoubleType(), True),
    StructField("std_intensity", DoubleType(), True)
])


def extract_features(image_path):

    try:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return (None, None, None, None)

        height, width = img.shape

        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        return (
            width,
            height,
            mean_intensity,
            std_intensity
        )

    except Exception:
        return (None, None, None, None)


feature_udf = udf(extract_features, schema)

# APPLY FEATURE EXTRACTION
df = df.withColumn(
    "features",
    feature_udf("image_path")
)

# FLATTEN STRUCT
processed_df = (
    df
    .withColumn("width", df.features.width)
    .withColumn("height", df.features.height)
    .withColumn("mean_intensity", df.features.mean_intensity)
    .withColumn("std_intensity", df.features.std_intensity)
    .drop("features")
)


processed_df.show(5)

(
    processed_df.write
    .mode("overwrite")
    .parquet(OUTPUT_DIR)
)

print(f"Saved features to {OUTPUT_DIR}")
spark.stop()
