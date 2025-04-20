### Multimodel AI Cyberattack Detection System 

This project is an AI-powered cyberattack detection system that fuses structured network traffic data and textual alert messages to enhance accuracy and reduce false positives. It includes a FastAPI backend and a Streamlit dashboard, fully containerized with Docker.

---

## Project Overview

- **Model 1**: A Random Forest classifier trained on structured features from the CICIDS2017 dataset.  
   [CICIDS2017 Dataset](https://drive.google.com/drive/folders/14N85Sa08HkzvgpNdVI4jJQUw879wheCj?usp=drive_link)  
   [Cleaned & Preprocessed Version used to train model 1 on](https://drive.google.com/drive/folders/1qhSGynYApq1ts2ZLMGSdoQA_IVJnWGb4?usp=drive_link)

- **Model 2**: A Logistic Regression classifier trained on TF-IDF vectorized IDS-style alert messages.  
   [Alerts Dataset](https://drive.google.com/drive/folders/1lBys20y8nrd_R6xTPNa_pcbjKPnEdKd4?usp=drive_link)  
   [Original Source Dataset used to generate the Alerts Dataset](https://drive.google.com/drive/folders/1FQ6SO3C6FxwtC2_d6NJQ29ZPHfxMoF52?usp=drive_link)


- **Fusion Model**: Combines both models’ predictions with a dynamic threat scoring system and recommended actions.
- **GUI Dashboard**: Built with Streamlit for interactive attack simulation and threat analysis.
- **API**: FastAPI-based RESTful service supporting traffic, alert, and fusion prediction endpoints.

---

Run with Docker
## Requirements
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)


## Quick Start

Clone the project:

git clone https://github.com/ayman876/multimodel-ai-detection.git

cd multimodel-ai-detection

## Run the containers
docker compose up --build

## Access
API Docs: http://localhost:8000/docs

Dashboard UI: http://localhost:8501


## Contents
api/ – FastAPI backend (prediction endpoints)

dashboard/ – Streamlit UI for interacting with the models

models/ – Pretrained model files (.joblib)

docker-compose.yml – Docker orchestration file


