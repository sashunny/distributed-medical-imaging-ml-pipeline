# Distributed Medical Imaging ML Pipeline

A scalable machine learning pipeline for MRI image processing built using PySpark, Spark SQL, Scikit-Learn, and Docker. The project demonstrates end-to-end ML engineering workflows including data ingestion, distributed feature extraction, batch inference, model evaluation, and analytics on medical imaging dataset (the basic example uses Brain Tumor MRI scans but can easlily be extended).

## Overview

This project was developed to explore scalable machine learning processing pipelines for medical imaging workloads. The pipeline automates data ingestion, feature engineering, model training, batch inference, and analytics using distributed data processing principles.

The implementation focuses on ML engineering and data pipeline design rather than state-of-the-art medical image classification performance.

---

## Architecture

```text
MRI Images
    │
    ▼
PySpark Ingestion
    │
    ▼
Parquet Storage
    │
    ▼
Distributed Feature Extraction
    │
    ▼
Random Forest Training
    │
    ▼
Batch Inference
    │
    ▼
Spark SQL Analytics
    │
    ▼
Evaluation & Reporting
```

---

## Dataset

Dataset used:

**Brain MRI Images for Brain Tumor Detection**

The project uses the [Kaggle Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) for training and evaluation.

Classes:

- yes → Tumor Present
- no → Tumor Absent

Dataset structure:

```text
data/MRI_brain_tumor/brain_tumor_dataset/
├── yes/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## Technology Stack

### Languages

- Python

### Distributed Data Processing

- PySpark
- Spark SQL
- Parquet

### Machine Learning

- Scikit-Learn
- Random Forest Classifier

### Data Processing

- OpenCV
- NumPy
- Pandas

### Deployment

- Docker

---

## Pipeline Components

### 1. Data Ingestion

`ingest.py`

Automatically scans MRI image directories and creates a distributed metadata dataset.

Generated metadata:

- image_id
- image_path
- label
- file_size_bytes

Output:

```text
data/MRI_brain_tumor/parquet/mri_metadata
```

---

### 2. Feature Engineering

`preprocess.py`

Extracts image-level features using distributed Spark processing.

Generated features:

- width
- height
- mean_intensity
- std_intensity

Output:

```text
data/MRI_brain_tumor/parquet/mri_features
```

---

### 3. Model Training

`train.py`

Trains a Random Forest classifier using extracted MRI features.

Features used:

- width
- height
- mean_intensity
- std_intensity

Output:

```text
models/random_forest.pkl
```

---

### 4. Batch Inference

`inference.py`

Simulates production batch scoring by applying the trained model across the feature dataset.

Generated outputs:

- prediction
- probability

Output:

```text
data/MRI_brain_tumor/parquet/predictions
```

---

### 5. Analytics & Evaluation

`analytics.py`

Computes model evaluation metrics and generates visual reports.

Metrics:

- Accuracy
- Recall
- F1 Score
- ROC-AUC

Visualizations:

- Confusion Matrix
- ROC Curve
- Class Distribution

Output:

```text
analytics/
├── confusion_matrix.png
├── roc_curve.png
└── class_distribution.png
```

---

## Example Spark SQL Analytics

The pipeline supports SQL-based analytics on prediction outputs.

Example:

```sql
SELECT
    label,
    COUNT(*) AS total
FROM predictions
GROUP BY label;
```

Example:

```sql
SELECT
    label,
    AVG(mean_intensity) AS avg_intensity
FROM predictions
GROUP BY label;
```

---

## Repository Structure

```text
distributed-medical-imaging-ml-pipeline/

├── data/
│   ├── MRI_brain_tumor/
|   |   ├── parquet/
│   │   |   ├── mri_metadata/
│   │   |   ├── mri_features/
│   │   |   └── predictions/
|
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── inference.py
│   ├── analytics.py
│
├── models/
│   └── random_forest.pkl
|
├── analytics/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── class_distribution.png
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── ReadMe.md
```

---

## Running the Pipeline

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Execute Pipeline

```bash
python ingest.py
```

```bash
python preprocess.py
```

```bash
python train.py
```

```bash
python batch_inference.py
```

```bash
python analytics.py
```

---

## Docker

Build:

```bash
docker build -t distributed-mri-pipeline .
```

Run:

```bash
docker run distributed-mri-pipeline
```

---

## Key ML Engineering Concepts Demonstrated

- Distributed data processing using PySpark
- Spark DataFrames and Spark SQL
- Parquet-based storage
- Feature engineering pipelines
- Batch inference workflows
- Model evaluation and reporting
- Containerized deployment with Docker
- End-to-end machine learning pipeline design

---

## Future Improvements

- Deep learning inference using Vision Transformers (ViT)
- Distributed model training with Spark MLlib
- MLflow experiment tracking
- Data validation and monitoring
- AWS S3 integration
- FastAPI model serving
- Kubernetes deployment

---
