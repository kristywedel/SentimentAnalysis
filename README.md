# Sentiment Analysis Project

## Overview
This project demonstrates the development and deployment of a sentiment analysis application using machine learning techniques and NLP models. The workflow includes data preparation, model training, evaluation, and deployment as a FastAPI application for real-time sentiment prediction.

---

## Project Workflow

### 1. **Introduction to Sentiment Analysis**
Sentiment analysis is a natural language processing (NLP) technique used to classify text as positive, negative, or neutral. It provides actionable insights for applications like customer feedback analysis and social media monitoring.

### 2. **Dataset Overview**
- **File**: `customer_feedback.csv`
- **Columns**:
  - `Review_ID`: Unique identifier for each review.
  - `Review_Text`: Customer review text.
  - `Sentiment`: Sentiment label (Positive, Neutral, Negative).
- **Characteristics**:
  - Includes intentional duplicates and missing values to simulate real-world data cleaning challenges.

### 3. **Steps Covered**
#### Data Preparation:
1. Remove duplicate rows based on the `Review_Text` column.
2. Drop rows with missing values in `Review_Text` or `Sentiment` columns.
3. Preprocess text:
   - Convert to lowercase.
   - Remove punctuation and stop words.
   - Apply lemmatization.

#### Model Training and Evaluation:
1. Train the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face using TensorFlow.
2. Group Positive and Neutral sentiments for simplified classification.
3. Evaluate the model with metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
4. Optimize hyperparameters (e.g., learning rate, batch size).

#### Deployment:
1. Deploy the model using FastAPI:
   - Create a REST API endpoint (`http://127.0.0.1:8000/predict`).
   - Enable real-time sentiment predictions.
2. Test the API with a sample client script.

---

## Installation

### Prerequisites
- **Conda**:
  - [Miniconda (lightweight option)](https://www.anaconda.com/download)
  - [Anaconda (includes more tools)](https://www.anaconda.com/download)
  - Alternatively, you can install Python directly, but this won’t have the dependencies pre-loaded.  
  - [Watch: How to Install Conda](https://www.youtube.com/watch?v=wwKAFw2Dwz0)
- [Cursor](https://www.cursor.com/) (Requires creating a free account)  
  - [Watch: How to Install Cursor](https://www.youtube.com/watch?v=L_A2P1KJYpw)

### Setup Steps
1. **Download the GitHub repository and extract it:**
   - GitHub link: [https://bit.ly/3PIK4RU](https://bit.ly/3PIK4RU)  
   - [Watch: How to Download GitHub Files and Set Up Conda](https://www.youtube.com/watch?v=Fv_tDvJuJpk)

   Alternatively, use Git to clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Conda Env Setup:**
   - Open a command prompt and navigate (cd) to the repository directory
   ```bash
   conda env create -f environment.yml
   conda activate workshop_env
   ```
3. **Cursor Setup:**
   - [Watch: How to Set Up Cursor](https://youtu.be/5ISWpw4bSVE)

---

## Usage

### Running the FastAPI Application
1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
2. Access the API at:
   - Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Predict: `http://127.0.0.1:8000/predict`

### Testing the API
1. Use `test_api.py` to send sample requests and validate predictions:
   ```bash
   python test_api.py
   ```

---

## Key Takeaways
- **Skills Gained**:
  - Data cleaning and preprocessing for NLP tasks.
  - Fine-tuning pre-trained transformer models.
  - Deploying a real-time machine learning application with FastAPI.
- **Insights**:
  - High-quality data preparation significantly impacts model performance.
  - Grouping sentiments simplifies classification but reduces granularity.
  - Deployment allows practical application for real-time sentiment analysis.

---

## Future Enhancements
1. Implement advanced models (e.g., DeBERTa, RoBERTa) for improved accuracy.
2. Explore hyperparameter optimization using tools like Optuna.
3. Add data augmentation for more robust training.
4. Expand the API to handle additional text analysis tasks.

---

## License
This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/)
