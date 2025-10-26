# picnic-topic-prediction

## Overview
This repository contains a containerized machine learning pipeline for multi-class classification of news articles using TF-IDF and LightGBM, with hyperparameter tuning via Optuna and experiment tracking with MLflow. It also contains a notebook for zero-shot classification, which serves as an alternative approach to this problem. 

## Project Description
This project is to build a multi-class classification model to predict news categories. The AG News dataset is used with 120k rows of training data and 7.6k rows of test data with perfectly balanced class distribution.

Two models have been explored in this project. They are:
- A scikit-learn pipeline with TF-IDF vectorizer and LightGBM
- A pre-trained zero-shot classifier

I have not experimented with prompting LLM directly because it is less reliable and requires more work to validate the output, while a zero-shot classifier also leverages the power of deep learning and has a more stable performance.

Due to time and hardware constraints, I did not run the training and evaluation pipeline on the full data. I did the following two experiments.
- TF-IDF + LightGBM (10k training, 5k test, 10 trials for HP tuning) - 85% training accuracy, 86% test accuracy
- Zero-shot classifier (very basic model with 200M parameters, 100 test) - 84% test accuracy

If you run the pipeline following the instructions in the Quickstart session, you will be able to access the MLFlow UI via http://localhost:5050/ and see the training and test accuracies, macro F1 scores, confusion matrices and feature importance plot showing the top 20 words used in the model.

If time allows, I would like to make the following improvements:
- Optimize the lemmatizer to speed it up (it is currently declared but not used due to speed)
- Improve the tokenizer and enrich the stop word list based on training data
- Use SHAP to identify top words for each category (e.g. 'oil' may very often be associated with world news) and check if they make sense
- Add all cross-validation trials to MLFlow artifacts so we can see not only the best run but all
- Make it easier to explore test results on MLflow so we can see where the model gets it wrong
- Experiment with more models and more hyperparameters
- Experiment with larger zero-shot classification models
- Fine-tune zero-shot classification models on AG News data

## Quick Start

### Using Docker (recommended)
```
docker compose up --build
```


## Project Structure
```
├── data/                  # Training and test splits in Parquet format
├── notebook/              # Demonstration notebook for zero‑shot classification
├── resource/              # Auxiliary assets (e.g. NLTK wordnet)
├── src/picnic_topic_prediction/
│   ├── config.py          # Configuration and hyper‑parameter search 
│   ├── train.py           # Training pipeline with Optuna and MLflow 
│   ├── eval.py            # Evaluation routine for the test set
│   ├── utils.py           # Helpers for loading data and plotting confusion matrices
├── main.py                # Entry point that orchestrates training and evaluation
├── pyproject.toml         # Project metadata and dependency list
├── Dockerfile, compose    # Optional containerisation for reproducible environments
└── README.md              # You are here
```