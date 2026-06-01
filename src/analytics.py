from pyspark.sql import SparkSession

PREDICTIONS_PATH = "../data/MRI_brain_tumor/parquet/predictions"

spark = (
    SparkSession.builder
    .appName("MRIAnalytics")
    .getOrCreate()
)

df = spark.read.parquet(
    PREDICTIONS_PATH
)

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
