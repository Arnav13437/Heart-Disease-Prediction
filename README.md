# Heart Disease Risk Predictor

An AI-assisted full-stack web application that predicts the likelihood of heart disease based on 13 clinical parameters. The system uses a Logistic Regression machine learning model trained on the Cleveland Heart Disease dataset.

## Architecture

- **Backend**: Python, FastAPI, Scikit-Learn
- **Frontend**: React.js, Vite, Vanilla CSS 

## Features
- **Machine Learning Inference**: Trains a Logistic Regression classification model on startup and serves predictions via a REST API.
- **Modern UI**: Reactive, responsive, dark-themed glowing user interface designed with React and CSS animations.

## Setup Instructions

### 1. Backend Setup (FastAPI & ML Model)

Ensure you are in the root directory and create a Python virtual environment:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

Run the FastAPI ML server:
```bash
python heartdisease.py
```
*The backend will train the model instantly and be available at `http://localhost:8000`.*

### 2. Frontend Setup (React/Vite)

Open a new terminal window and navigate to the `frontend` directory:
```bash
cd frontend
```

Install Node dependencies:
```bash
npm install
```

Start the Vite development web server:
```bash
npm run dev
```
*The web interface will be available at `http://localhost:5173`. Input your parameters and generate real-time inferences!*

## Disclaimer
⚠️ This is an AI-assisted research and educational project, not a medical diagnostic tool. Please always consult a qualified cardiologist for medical advice.
