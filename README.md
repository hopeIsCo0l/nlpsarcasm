# Sarcasm Detection System

A comprehensive NLP-based sarcasm detection system for headlines using machine learning and deep learning approaches.

## Project Overview

This project implements a sarcasm detection system that can classify news headlines as sarcastic or non-sarcastic. The system uses the Sarcasm Headlines Dataset and provides both traditional machine learning and deep learning approaches.

## Project Structure

```
NLP-CapStone/
├── data/                           # Dataset files
│   ├── Sarcasm_Headlines_Dataset.json
│   └── sarcasm_dataset_cleaned.csv
├── models/                         # Trained model files
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                           # Source code
│   ├── api/                       # API endpoints
│   ├── preprocessing/             # Text preprocessing modules
│   ├── models/                    # Model training and inference
│   └── utils/                     # Utility functions
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Features

- **Data Exploration**: Comprehensive analysis of the sarcasm dataset
- **Text Preprocessing**: Advanced NLP preprocessing pipeline
- **Multiple Models**: Traditional ML and deep learning approaches
- **Model Evaluation**: Comprehensive performance analysis
- **API Endpoint**: FastAPI-based prediction service
- **Scalable Architecture**: Modular design for easy extension

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd NLP-CapStone

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The dataset is already included in the `data/` folder. Run the data exploration notebook to understand the dataset:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Model Training

Follow the notebooks in order:
1. `01_data_exploration.ipynb` - Understand the dataset
2. `02_text_preprocessing.ipynb` - Preprocess the text data
3. `03_model_training.ipynb` - Train various models
4. `04_model_evaluation.ipynb` - Evaluate model performance

### 4. API Deployment

```bash
# Start the FastAPI server
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Using the API

Once the API is running, you can make predictions:

```python
import requests

# Example API call
url = "http://localhost:8000/predict"
data = {"headline": "Scientists discover that coffee is actually good for you"}
response = requests.post(url, json=data)
print(response.json())
```

### Using the Models Directly

```python
from src.models.sarcasm_detector import SarcasmDetector

# Load trained model
detector = SarcasmDetector()
detector.load_model('models/best_model.pkl')

# Make prediction
prediction = detector.predict("Your headline here")
print(f"Is sarcastic: {prediction}")
```

## Model Performance

The system includes multiple model approaches:

1. **Traditional ML Models**:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
   - Naive Bayes

2. **Deep Learning Models**:
   - LSTM
   - CNN
   - BERT-based models
   - Transformer models

## API Documentation

Once the API is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Endpoints

- `POST /predict` - Predict sarcasm for a given headline
- `GET /health` - Health check endpoint
- `GET /model-info` - Get information about the loaded model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sarcasm Headlines Dataset creators
- Open source NLP community
- FastAPI and scikit-learn communities
- Dr. Fantahun Bogale
