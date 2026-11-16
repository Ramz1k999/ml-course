# Cardiovascular Disease Risk Prediction

Midterm project for [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.

A machine learning service that estimates the probability of cardiovascular disease from the results of a routine medical check-up. The model is trained on 70,000 patient records and served through a Flask web service packaged in Docker.

> **Disclaimer:** this is an educational project. The model is not a medical device and must not be used for diagnosis or treatment decisions.

## Table of contents

- [Problem description](#problem-description)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Exploratory data analysis](#exploratory-data-analysis)
- [Models and results](#models-and-results)
- [How to run the project](#how-to-run-the-project)
- [Using the web service](#using-the-web-service)
- [Docker](#docker)
- [Cloud deployment](#cloud-deployment)

## Problem description

Cardiovascular disease is one of the leading causes of death worldwide, and early detection greatly improves outcomes. At a routine check-up a clinic collects simple measurements (age, height, weight, blood pressure, cholesterol and glucose levels, lifestyle habits), but doctors have limited time to review every patient in detail.

**Goal:** predict whether a patient has cardiovascular disease (binary classification) and return the probability, so that the result can be used as a screening aid.

**How the solution would be used:** a clinic's registration or check-up system sends a patient's measurements to the web service and receives a risk probability. Patients with a high probability can be prioritized for additional examinations (ECG, blood tests, a cardiologist's appointment), while the doctor still makes the final decision.

The target variable is `cardio` (1 = disease present, 0 = absent). The classes are balanced (about 50/50), so ROC AUC is used as the main metric.

## Dataset

[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle (author: Svetlana Ulianova). It contains 70,000 records with 11 features and the target.

The file is included in the repository root: `cardio_train.csv` (values separated by `;`).

| Feature | Description |
|---|---|
| `age` | Age in days (converted to years in the project) |
| `gender` | 1 = women, 2 = men |
| `height` | Height, cm |
| `weight` | Weight, kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | Glucose: 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | Smoker (0/1) |
| `alco` | Alcohol intake (0/1) |
| `active` | Physical activity (0/1) |
| `cardio` | **Target:** presence of cardiovascular disease (0/1) |

## Project structure

```
.
├── README.md
├── notebook.ipynb        # data cleaning, EDA, feature importance, model selection and tuning
├── train.py              # trains the final model and saves it to model.bin
├── predict.py            # Flask web service that loads the model and serves predictions
├── test_predict.py       # example request to the running service
├── model.bin             # saved DictVectorizer + XGBoost model
├── requirements.txt      # dependencies
├── Dockerfile            # container for the web service
├── .dockerignore
└── cardio_train.csv      # dataset
```

## Exploratory data analysis

The full analysis is in `notebook.ipynb`. Main findings:

**Data cleaning**
- No missing values, but the data contains impossible values: negative blood pressure, `ap_hi` up to 16,020 and `ap_lo` up to 11,000, height of 55 cm, weight of 10 kg, BMI from 3.5 to almost 300.
- Records were removed when: `ap_hi` was outside 70–250, `ap_lo` outside 40–150, `ap_hi <= ap_lo`, height outside 120–220 cm, weight outside 30–200 kg, or BMI outside 15–60.
- About 2% of the rows were removed, leaving roughly 68,600 records.
- Age was converted from days to years, and a `bmi` feature was added (`weight / height²`).

**Target variable**
- Classes are balanced (about 50.5% healthy and 49.5% with disease), so no resampling is needed.

**Relationship between features and the target**
- The share of patients with disease grows strongly with age: from about 23% in the group up to 40 years to about 67% in the group over 60.
- It grows with cholesterol level: about 44% at normal level versus 76% at the highest level.
- Glucose has a moderate effect (about 48% to 62%), physical activity a weak one.

**Feature importance**
- Correlation with the target (numeric features): `ap_hi` 0.43, `ap_lo` 0.34, `age` 0.24, `bmi` 0.19, `weight` 0.18, `height` -0.01.
- Mutual information (categorical features): `cholesterol` 0.025, `gluc` 0.004, `active` 0.0007, `smoke` 0.0001, `alco` and `gender` about 0.
- Blood pressure, age and cholesterol are the strongest signals. Lifestyle features (smoking, alcohol, activity) and gender carry almost no information in this dataset, possibly because they are self-reported.

## Models and results

The data was split into train / validation / test (60% / 20% / 20%). Models were compared by ROC AUC on the validation set, and parameters were tuned for each model.

| Model | Tuned parameters | Best validation AUC |
|---|---|---|
| Logistic regression (with standard scaling) | `C` | 0.788 |
| Random forest | `max_depth`, `min_samples_leaf` | 0.797 |
| **XGBoost** | `max_depth`, `learning_rate` | **0.798** |

Logistic regression is not sensitive to `C` on scaled features. The differences between the best random forest and XGBoost configurations are very small, and the quality is limited by the data itself. XGBoost with `max_depth=4`, `learning_rate=0.05`, `n_estimators=200` was selected as the final model.

The final model was trained on train + validation and evaluated once on the test set: **ROC AUC = 0.796**, which is very close to the validation score, so there are no signs of overfitting.

## How to run the project

Requirements: Python 3.12 or newer, Git. (Docker is needed only for the container part.)

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/cardio-risk-prediction.git
cd cardio-risk-prediction
```

**2. Create and activate a virtual environment**

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Reproduce the analysis (optional)**

Open `notebook.ipynb` in Jupyter or VS Code and run all cells. The dataset is already in the repository root.

**5. Train the model**

```bash
python train.py
```

This cleans the data, trains the final XGBoost model, prints the test AUC (about 0.796) and saves the model with its `DictVectorizer` to `model.bin`.

**6. Start the web service**

```bash
python predict.py
```

The service listens on `http://localhost:9696`.

## Using the web service

The service has one endpoint: `POST /predict`. It accepts a JSON object with the patient's data (age in **years**) and returns the probability of disease. BMI is computed by the service.

Example (`test_predict.py`):

```python
import requests

url = 'http://localhost:9696/predict'

patient = {
    'age': 55.0,
    'gender': 1,
    'height': 165,
    'weight': 85.0,
    'ap_hi': 140,
    'ap_lo': 90,
    'cholesterol': 3,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1,
}

print(requests.post(url, json=patient).json())
```

Run it in a second terminal while the service is running:

```bash
python test_predict.py
```

Response:

```json
{"cardio_probability": 0.841, "cardio_risk": true}
```

`cardio_risk` is `true` when the probability is at least 0.5.

## Docker

Build the image:

```bash
docker build -t cardio-prediction .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 cardio-prediction
```

The service is then available at `http://localhost:9696/predict`, and `python test_predict.py` works the same way as for the local run.

## Cloud deployment

The service is not deployed to a public URL in this repository, but it can be deployed to the cloud with any platform that builds Docker images. Below are the steps for [Render](https://render.com) (a free plan is available).

1. Push the repository to GitHub (it must contain `Dockerfile`, `predict.py`, `model.bin` and `requirements.txt`):

   ```bash
   git init
   git add .
   git commit -m "cardio risk prediction project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/cardio-risk-prediction.git
   git push -u origin main
   ```

2. On Render choose **New → Web Service → Build and deploy from a Git repository** and select the repository.
3. Set **Language** to **Docker** and the instance type to **Free**. Render finds the `Dockerfile` in the repository root and builds the image automatically.
4. If Render does not detect the port, set it to `9696`, which is the port used in the `Dockerfile`.
5. Click **Create Web Service**. After the build finishes, the service is available at `https://<service-name>.onrender.com`.
6. To test it, set `url = 'https://<service-name>.onrender.com/predict'` in `test_predict.py` and run `python test_predict.py`.

Note: on the free plan the service goes to sleep when idle, so the first request after a pause can take up to a minute.
