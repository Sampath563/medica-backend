# Medical Diagnosis Models

This directory should contain your trained machine learning models:

1. **best_medical_model_logistic_regression.pkl** - Your trained logistic regression model
2. **ensemble_medical_model.pkl** - Your trained ensemble model
3. **medical_scaler.pkl** - The scaler used for preprocessing numerical features
4. **symptom_vectorizer.pkl** - The TF-IDF vectorizer for symptoms text processing

## Setup Instructions

1. Copy your trained model files from `D:\medica\model\` to this directory
2. Ensure all four files are present before starting the backend server
3. The backend will automatically load these models on startup

## Model Files Expected

- `best_medical_model_logistic_regression.pkl` (50 KB)
- `ensemble_medical_model.pkl` (17.9 MB)
- `medical_scaler.pkl` (2 KB)
- `symptom_vectorizer.pkl` (3 KB)

## Usage

The backend API will use these models to:
- Vectorize symptom text using the TF-IDF vectorizer
- Scale numerical vital signs using the medical scaler
- Make predictions using both the logistic regression and ensemble models
- Return the most confident prediction along with all model results