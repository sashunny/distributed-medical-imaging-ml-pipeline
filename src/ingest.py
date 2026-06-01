from pathlib import Path
from pyspark.sql import SparkSession


DATASET_DIR = "../data/MRI_brain_tumor/brain_tumor_dataset"
OUTPUT_DIR =  "../data/MRI_brain_tumor/parquet/mri_metadata"

spark = (
    SparkSession.builder.appName("BrainMRIIngestion").getOrCreate()
)

# build records

records = []

dataset_path = Path(DATASET_DIR)

accepted_files_types = [".jpg",".jpeg",".png"]

for class_dir in dataset_path.iterdir():

    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for image_file in class_dir.glob("*"):

        if image_file.suffix.lower() not in accepted_files_types:
            continue

        records.append(
            {
                "image_id": image_file.stem,
                "image_path": str(image_file.resolve()),
                "label": label,
                "file_size_bytes": image_file.stat().st_size
            }
        )


# Create dataframe
df = spark.createDataFrame(records)

print("\nSchema")
df.printSchema()

print("\nSample Rows")
df.show(10, truncate=False)

print("\nClass Distribution")
df.groupBy("label").count().show()


# Write PARQUET
(
    df.write
    .mode("overwrite")
    .parquet(OUTPUT_DIR)
)

print(f"\nSaved parquet metadata to: {OUTPUT_DIR}")
spark.stop()
