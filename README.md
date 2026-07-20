# AI-Powered Semiconductor Wafer Defect Inspection System

An end-to-end deep learning system for automated semiconductor wafer defect classification using a custom Convolutional Neural Network (CNN).

This project demonstrates the complete lifecycle of an AI engineering solution, starting from dataset preparation and model development through API deployment, interactive dashboard development, database integration, explainability, and Docker containerization.

The system is designed to classify semiconductor wafer defect patterns from wafer map images and expose the trained model through a production-style REST API and Streamlit dashboard.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Business Problem](#business-problem)
* [Project Objectives](#project-objectives)
* [Dataset](#dataset)
* [Defect Categories](#defect-categories)
* [End-to-End Workflow](#end-to-end-workflow)
* [System Architecture](#system-architecture)
* [Model Development](#model-development)
* [Model Performance](#model-performance)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Main Features](#main-features)
* [API Endpoints](#api-endpoints)
* [Dashboard](#dashboard)
* [Database Integration](#database-integration)
* [Explainable AI](#explainable-ai)
* [Docker Deployment](#docker-deployment)
* [Local Installation](#local-installation)
* [Environment Configuration](#environment-configuration)
* [Running the Application](#running-the-application)
* [Testing the System](#testing-the-system)
* [Results](#results)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Author](#author)

---

## Project Overview

Semiconductor wafer inspection is an important quality-control process in semiconductor manufacturing.

During wafer fabrication, defects may occur due to issues such as process variation, contamination, equipment instability, material problems, or manufacturing inconsistencies. These defects can form recognizable spatial patterns on wafer maps.

Manual inspection of wafer maps can be:

* Time-consuming
* Inconsistent
* Difficult to scale
* Dependent on expert knowledge
* Vulnerable to human error

This project applies deep learning to automatically classify wafer defect patterns using the WM-811K wafer map dataset.

The final solution includes:

* Data cleaning and preprocessing
* Wafer map image transformation
* CNN model training
* Hyperparameter tuning
* Model evaluation
* FastAPI inference service
* Streamlit dashboard
* PostgreSQL database
* Explainable AI visualization
* Docker-based deployment

The objective is not only to train a machine learning model, but also to demonstrate how the model can be integrated into a complete AI application.

---

## Business Problem

In semiconductor manufacturing, detecting wafer defects early can help reduce production waste, improve yield, and support faster root-cause analysis.

A wafer map contains the test results of individual dies located on a semiconductor wafer. The spatial arrangement of failed dies may indicate specific manufacturing problems.

Traditional inspection methods often require engineers or operators to manually review these wafer maps.

A machine learning system can assist this process by:

1. Automatically detecting wafer defect patterns
2. Providing prediction confidence scores
3. Storing prediction history
4. Supporting faster quality-control decisions
5. Providing visual explanations for model predictions

This project simulates an AI-powered wafer inspection system that can be used as a foundation for a manufacturing quality-control application.

---

## Project Objectives

The main objectives of this project are:

* Build an image classification model for wafer defect patterns
* Clean and preprocess the WM-811K dataset
* Handle class imbalance in the dataset
* Compare model configurations
* Tune important model hyperparameters
* Evaluate the model using multiple classification metrics
* Develop a REST API for model inference
* Build an interactive dashboard for users
* Store prediction records in PostgreSQL
* Add explainability features
* Containerize the complete system using Docker
* Demonstrate a production-oriented AI engineering workflow

---

## Dataset

This project uses the:

**WM-811K Wafer Map Dataset**

The WM-811K dataset contains more than 800,000 wafer maps collected from real semiconductor manufacturing processes.

The dataset includes wafer maps with labelled defect patterns and wafer maps without defect labels.

For this project, only the relevant labelled samples were selected for model development.

The original dataset is not included in this repository due to:

* Large file size
* Repository storage limitations
* Dataset licensing and redistribution considerations

Users should obtain the dataset separately and place it in the appropriate local data directory.

---

## Defect Categories

The model is designed to classify wafer maps into multiple defect-pattern categories.

The dataset mapping used in this project includes classes such as:

* Center
* Donut
* Edge-Loc
* Edge-Ring
* Loc
* Near-Full
* Random
* Scratch
* None

Each class represents a different spatial distribution of failed dies on a wafer map.

The `None` category represents wafer maps without a recognized defect pattern.

---

## End-to-End Workflow

The complete project workflow is:

```text
Raw WM-811K Dataset
        │
        ▼
Data Exploration
        │
        ▼
Data Cleaning and Validation
        │
        ▼
Label Extraction
        │
        ▼
Wafer Map Preprocessing
        │
        ▼
Train, Validation and Test Split
        │
        ▼
CNN Model Development
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Model Artifact Export
        │
        ▼
FastAPI Inference Service
        │
        ▼
PostgreSQL Prediction Storage
        │
        ▼
Streamlit Dashboard
        │
        ▼
Docker Containerization
```

---

## System Architecture

The deployed application contains three main services:

### 1. Streamlit Dashboard

The Streamlit frontend allows users to:

* Upload wafer map images
* Select sample images
* Submit images for prediction
* View the predicted defect class
* View prediction confidence
* Display explainability outputs
* Review model information
* Interact with the prediction system

### 2. FastAPI Backend

The FastAPI service is responsible for:

* Loading the trained CNN model
* Receiving image upload requests
* Preprocessing input images
* Running model inference
* Returning prediction results
* Generating confidence scores
* Storing prediction history
* Exposing analytics and system endpoints

### 3. PostgreSQL Database

PostgreSQL stores application data such as:

* Prediction class
* Confidence score
* Prediction timestamp
* Input metadata
* Model information
* Inference records

The high-level architecture is:

```text
User
  │
  ▼
Streamlit Dashboard
  │
  │ HTTP Request
  ▼
FastAPI REST API
  │
  ├── CNN Model Inference
  ├── Image Preprocessing
  ├── Prediction Confidence
  └── Explainability
  │
  ▼
PostgreSQL Database
```

---

## Model Development

A custom Convolutional Neural Network was developed to classify wafer defect images.

The model development process included:

### Data Cleaning

The cleaning process involved:

* Removing invalid samples
* Filtering missing labels
* Verifying wafer map dimensions
* Converting labels into consistent class names
* Removing unusable records
* Validating label distribution
* Checking class imbalance

### Image Preprocessing

Wafer maps were converted into a format suitable for CNN training.

The preprocessing pipeline included:

* Converting wafer map arrays into images
* Resizing images to a consistent input size
* Converting the input into tensors
* Applying normalization
* Encoding target labels
* Preparing batch loaders

### Dataset Splitting

The dataset was separated into:

* Training set
* Validation set
* Test set

The test data was kept separate from model training and tuning to provide a more reliable final evaluation.

### Class Imbalance

The WM-811K dataset contains an unequal number of samples for each defect class.

This is important because a model may achieve high accuracy while performing poorly on minority classes.

To address this issue, the project evaluated the model using metrics beyond standard accuracy, including:

* Balanced accuracy
* Macro F1 score
* Per-class precision
* Per-class recall
* Confusion matrix
* Macro AUC

### Hyperparameter Tuning

The model was improved by tuning selected parameters such as:

* Learning rate
* Batch size
* Number of convolutional filters
* Dropout rate
* Number of epochs
* Optimizer configuration
* Weight decay

The final model configuration was selected based on validation performance.

---

## Model Performance

The final CNN model achieved the following results:

| Metric            |       Score |
| ----------------- | ----------: |
| Accuracy          |  **97.23%** |
| Balanced Accuracy |  **89.32%** |
| Macro F1 Score    |  **86.72%** |
| Macro AUC         | **0.99671** |

### Metric Explanation

#### Accuracy

Accuracy measures the percentage of all predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

The final accuracy of 97.23% indicates that the model correctly classified most wafer map samples.

#### Balanced Accuracy

Balanced accuracy calculates the average recall across all classes.

This metric is especially important because the dataset contains class imbalance.

The balanced accuracy of 89.32% shows that the model performs strongly across both majority and minority classes.

#### Macro F1 Score

Macro F1 calculates the F1 score for every class and then gives equal weight to each class.

The final Macro F1 score of 86.72% indicates good overall performance across the different wafer defect categories.

#### Macro AUC

Macro AUC evaluates how well the model separates each class from the others.

The Macro AUC score of 0.99671 indicates excellent class-separation capability.

---

## Technology Stack

### Machine Learning

* Python
* PyTorch
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib

### Backend

* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy

### Frontend

* Streamlit

### Database

* PostgreSQL

### Deployment

* Docker
* Docker Compose

### Development Tools

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

---

## Project Structure

```text
.
├── app/
│   ├── api/
│   │   └── routes/
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   │
│   ├── models/
│   │   └── cnn.py
│   │
│   ├── services/
│   │   ├── inference.py
│   │   └── preprocessing.py
│   │
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── custom_cnn_best.pt
│   └── label_mapping.json
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── docs/
│   └── images/
│
├── outputs/
│
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The exact structure may differ slightly depending on the final implementation.

---

## Main Features

### Automated Wafer Defect Classification

The system accepts a wafer map image and predicts the most likely defect category.

### Prediction Confidence

The API returns a confidence score together with the predicted class.

### REST API

The trained model is exposed through FastAPI, making it possible to integrate the classifier with other software systems.

### Interactive Dashboard

The Streamlit interface provides a user-friendly way to test the model without writing code.

### Prediction History

Prediction records can be stored in PostgreSQL for future analysis.

### Explainability

The system provides visual explanations to help users understand which regions influenced the prediction.

### Containerized Deployment

FastAPI, Streamlit, and PostgreSQL are configured as separate Docker services.

---

## API Endpoints

The API documentation can be accessed through Swagger UI:

```text
http://localhost:8000/docs
```

Example API functions may include:

| Endpoint       | Method | Description                                 |
| -------------- | ------ | ------------------------------------------- |
| `/health`      | GET    | Check API status                            |
| `/predict`     | POST   | Upload a wafer map and receive a prediction |
| `/predictions` | GET    | Retrieve prediction history                 |
| `/analytics`   | GET    | Retrieve prediction analytics               |
| `/model-info`  | GET    | Retrieve model information                  |

The exact endpoint names may vary depending on the implementation.

---

## Dashboard

The Streamlit dashboard is available at:

```text
http://localhost:8501
```

The dashboard allows users to:

* Upload wafer map images
* Preview uploaded images
* Send images to the prediction API
* View predicted defect classes
* View confidence scores
* Display explanation heatmaps
* View model performance information
* Access prediction analytics

Recommended screenshots for the repository:

```text
docs/images/dashboard.png
docs/images/prediction-result.png
docs/images/swagger-api.png
docs/images/docker-running.png
docs/images/confusion-matrix.png
docs/images/system-architecture.png
```

Screenshots can be displayed in this README using:

```markdown
![Dashboard](docs/images/dashboard.png)
```

---

## Database Integration

The system uses PostgreSQL to store prediction data.

The database service is defined in `docker-compose.yml`.

Example information stored for each prediction may include:

* Prediction ID
* File name
* Predicted class
* Confidence score
* Prediction date
* Inference duration
* Model version

The database supports future analytics such as:

* Number of predictions by class
* Most frequently detected defect
* Average confidence score
* Prediction activity over time
* Model usage monitoring

---

## Explainable AI

Deep learning models are often considered black-box systems because their internal decision-making process is difficult to interpret.

This project includes explainability visualization to indicate which regions of the wafer map contributed to the model prediction.

An explanation heatmap can help:

* Validate whether the model focuses on relevant wafer regions
* Identify suspicious model behavior
* Improve user confidence
* Support engineering investigation
* Detect possible dataset bias

Explainability does not prove that a prediction is correct, but it provides additional context for reviewing the result.

---

## Docker Deployment

The system is containerized using Docker Compose.

The Docker environment contains:

* PostgreSQL database container
* FastAPI backend container
* Streamlit dashboard container

### Build and Run

From the root project directory, run:

```bash
docker compose up --build
```

For detached mode:

```bash
docker compose up -d --build
```

### Check Container Status

```bash
docker compose ps
```

Expected status:

```text
api         Up
dashboard   Up
database    Up (healthy)
```

### View Logs

View all logs:

```bash
docker compose logs -f
```

View API logs:

```bash
docker compose logs -f api
```

View dashboard logs:

```bash
docker compose logs -f dashboard
```

### Stop Containers

```bash
docker compose stop
```

### Start Existing Containers

```bash
docker compose start
```

### Stop and Remove Containers

```bash
docker compose down
```

The PostgreSQL data remains available because the project uses a named Docker volume.

---

## Local Installation

### Prerequisites

Ensure the following tools are installed:

* Python 3.11
* Git
* Docker Desktop
* Docker Compose

### Clone the Repository

```bash
git clone https://github.com/your-username/AI-Powered-Semiconductor-Wafer-Defect-Inspection-System.git
```

Move into the project folder:

```bash
cd AI-Powered-Semiconductor-Wafer-Defect-Inspection-System
```

### Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a local `.env` file based on `.env.example`.

Example:

```env
POSTGRES_DB=wm811k
POSTGRES_USER=wm811k
POSTGRES_PASSWORD=change_me
DATABASE_URL=postgresql+psycopg://wm811k:change_me@database:5432/wm811k
```

Do not upload the real `.env` file to GitHub.

The `.gitignore` file should include:

```gitignore
.env
.env.*
!.env.example
```

---

## Running the Application

### Run with Docker

```bash
docker compose up -d
```

Open the API documentation:

```text
http://localhost:8000/docs
```

Open the Streamlit dashboard:

```text
http://localhost:8501
```

### Run FastAPI Locally

```bash
python -m uvicorn app.main:app --reload
```

### Run Streamlit Locally

```bash
streamlit run frontend/streamlit_app.py
```

The actual frontend file name may differ depending on the project structure.

---

## Testing the System

After starting the Docker services, verify the container status:

```bash
docker compose ps
```

Test the API health endpoint:

```text
http://localhost:8000/docs
```

Then test the dashboard:

```text
http://localhost:8501
```

A successful test should confirm that:

* The API container is running
* The dashboard container is running
* PostgreSQL is healthy
* The model loads successfully
* Images can be submitted
* Predictions are returned
* Prediction records are stored

---

## Results

The completed project successfully demonstrates an end-to-end AI engineering workflow.

Key achievements include:

* Prepared and cleaned the WM-811K dataset
* Developed a custom CNN model
* Tuned model hyperparameters
* Evaluated the model using class-imbalance-aware metrics
* Achieved 97.23% test accuracy
* Achieved 89.32% balanced accuracy
* Achieved 86.72% Macro F1
* Achieved 0.99671 Macro AUC
* Created a FastAPI inference backend
* Built a Streamlit user interface
* Integrated PostgreSQL
* Added model explainability
* Containerized the complete system using Docker Compose

This project demonstrates skills in:

* Deep learning
* Computer vision
* Data preprocessing
* Model evaluation
* API development
* Database integration
* Software deployment
* Docker
* AI system design

---

## Limitations

Although the model achieved strong evaluation results, the current system has several limitations:

* The model is trained on the WM-811K dataset and may not generalize to wafer maps from other manufacturing environments.
* Real production wafer data may contain additional defect patterns.
* Dataset imbalance may still affect minority-class predictions.
* Explainability heatmaps should not be treated as definitive evidence.
* The system has not yet been tested under high API traffic.
* There is currently no automatic model retraining pipeline.
* The application has not yet been deployed to a public cloud environment.
* Authentication and user-management features are not included.
* The system is intended as a portfolio and research project, not a certified industrial inspection product.

---

## Future Improvements

Potential future developments include:

### Machine Learning

* Compare the custom CNN with ResNet, EfficientNet, and MobileNet
* Perform more extensive hyperparameter optimization
* Add data augmentation
* Improve minority-class performance
* Add uncertainty estimation
* Add anomaly detection
* Explore transfer learning

### MLOps

* Integrate MLflow
* Add model versioning
* Add experiment tracking
* Add data versioning
* Add automated retraining
* Add model monitoring
* Add drift detection
* Add CI/CD pipelines

### Deployment

* Deploy to AWS, Azure, or Google Cloud
* Add Kubernetes deployment
* Add HTTPS
* Add authentication
* Add rate limiting
* Add automated API testing
* Add centralized logging

### Application

* Add user accounts
* Add batch image upload
* Add downloadable prediction reports
* Add dashboard analytics
* Add production-line monitoring
* Add real-time camera inference
* Add support for multiple model versions

---

## Repository Files

Files recommended for inclusion in GitHub:

```text
app/
frontend/
artifacts/label_mapping.json
docs/
notebooks/
Dockerfile.api
Dockerfile.frontend
docker-compose.yml
requirements.txt
README.md
.env.example
.gitignore
LICENSE
```

The trained model file can be included if it is smaller than GitHub's file-size limit.

```text
artifacts/custom_cnn_best.pt
```

Files that should not be uploaded:

```text
.env
venv/
.venv/
pythoTESt/
__pycache__/
dataset/
datasets/
raw_data/
postgres_data/
inspection_data/
*.db
*.sqlite
*.log
```

---

## Disclaimer

This project was developed for educational, research, and portfolio purposes.

It is not intended to replace certified semiconductor manufacturing inspection systems or professional engineering review.

Predictions should be validated before being used in real manufacturing decisions.

---

## Author

**Muhammad Amjad Zakwan**

Bachelor of Computer Science with Honours in Data Science

Career interests:

* AI Engineering
* Machine Learning Engineering
* Data Science
* Computer Vision
* MLOps

---

## Project Status

```text
Dataset Preparation          Completed
Data Cleaning                Completed
Model Training               Completed
Hyperparameter Tuning        Completed
Model Evaluation             Completed
FastAPI Backend              Completed
Streamlit Dashboard          Completed
PostgreSQL Integration       Completed
Explainable AI               Completed
Docker Deployment            Completed
Cloud Deployment             Planned
CI/CD Pipeline               Planned
Model Monitoring             Planned
```

---

## License

This project may be distributed under the MIT License.

Add a `LICENSE` file to the repository before selecting a license for public distribution.

---

## Acknowledgements

* WM-811K wafer map dataset
* PyTorch open-source community
* FastAPI open-source community
* Streamlit open-source community
* PostgreSQL open-source community
* Docker open-source ecosystem
