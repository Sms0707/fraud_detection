🛡️ Cloud-Native Fraud Detection & Concept Drift Pipeline

An enterprise-grade, end-to-end machine learning pipeline demonstrating dynamic fraud detection, cost-matrix optimization, and automated concept drift monitoring using AWS S3 emulation.

Built as a prototype for the AWS Student Builder Group, this project showcases how to decouple cloud storage from ML compute while handling real-world, highly imbalanced financial datasets.

🌟 Key Features & Architecture

☁️ Cloud Data Lake Integration (LocalStack & AWS S3):
Storage and compute are strictly decoupled. The pipeline uses boto3 to dynamically ingest data from external APIs, persist it into a localized mock AWS S3 bucket (vit-fraud-data-lake-s3), and fetch it on demand for model training.

📥 Dynamic Data Ingestion (Kaggle API):
To bypass GitHub file-size limits, the script seamlessly integrates with the Kaggle API to dynamically pull the authoritative, 150MB+ ULB Credit Card Fraud Dataset directly into the cloud pipeline.

⚖️ Extreme Class Imbalance Handling:
The dataset features a severe imbalance (~0.17% fraud). The pipeline evaluates models using PR-AUC (Precision-Recall Area Under Curve) rather than ROC-AUC to prevent True Negative distortion.

🤖 Dual-Model ML Benchmarking:
Compares a robust supervised tree-based model (HistGradientBoosting) against an unsupervised anomaly detection algorithm (Isolation Forest) to evaluate performance on rare events.

💰 Business Cost-Matrix Optimization:
Translates ML metrics into business value. The pipeline calculates the optimal decision threshold by weighing the monetary impact of False Positives ($15 customer friction/support cost) against False Negatives ($250 direct chargeback loss).

📉 Automated Concept Drift Monitoring:
Simulates live incoming transaction batches and applies the Kolmogorov-Smirnov (KS-test). If statistical drift is detected (p-value < 0.05), the pipeline automatically triggers a retraining loop to adapt to evolving fraud patterns.

⚙️ Tech Stack

Cloud Infrastructure: LocalStack (AWS Emulation), Boto3 (AWS SDK)

Machine Learning: Scikit-Learn (HistGradientBoosting, Isolation Forest)

Statistical Diagnostics: SciPy (KS-Test)

Data Engineering: Pandas, NumPy, Kaggle API

🚀 Quick Start Guide for Evaluators

1. Prerequisites

Docker: Required to run the LocalStack AWS emulator.

Kaggle Account: Required to fetch the dataset via API.

2. Start the Mock AWS S3 Cloud

Run this command in your terminal to spin up the local AWS S3 environment:

docker run --rm -it -p 4566:4566 localstack/localstack:3.8.0


3. Setup the Project

Clone this repository and install the required dependencies:

git clone <your-github-repo-url>
cd fraud_pipeline
pip install -r requirements.txt


4. Secure Credential Configuration (Kaggle API)

To fetch the real-world dataset securely, this script requires Kaggle API credentials.
Set them as environment variables in your terminal before running the script, or create a .env file (never commit this file to GitHub):

Windows (Command Prompt):

set KAGGLE_USERNAME=your_username

set KAGGLE_KEY=your_api_key


Mac/Linux:

export KAGGLE_USERNAME="your_username"

export KAGGLE_KEY="your_api_key"


5. Execute the Pipeline

Run the main script and follow the interactive terminal prompts:

python main.py
