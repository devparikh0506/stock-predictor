# Stock Sentiment Analysis and Price Prediction

This project implements a sentiment analysis model for stock market prediction using news headlines and historical stock data.

## Project Overview

The project combines sentiment analysis of financial news headlines with historical stock price data to predict future stock price movements. It demonstrates data preprocessing, feature engineering, model training, and evaluation.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- yfinance

## Dataset

The project uses two main datasets:
1. News headlines dataset (Combined_News_DJIA.csv)
2. Historical stock price data fetched using the yfinance library

## Project Structure

The Jupyter notebook contains the following main sections:

1. **Data Loading and Preprocessing**: 
   - Loading news headlines and stock price data
   - Text preprocessing and sentiment analysis
   - Feature engineering

2. **Exploratory Data Analysis**:
   - Visualizing stock price trends
   - Analyzing sentiment distribution

3. **Model Development**:
   - Preparing features and target variables
   - Splitting data into training and testing sets
   - Implementing and training multiple models:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - Long Short-Term Memory (LSTM) neural network

4. **Model Evaluation**:
   - Comparing model performances using accuracy and other metrics
   - Visualizing prediction results

## Usage

1. Ensure all required libraries are installed.
2. Open and run the Jupyter notebook `Stock-Sentiment-Analysis.ipynb`.
3. Follow the notebook cells to preprocess data, train models, and evaluate results.

## Key Features

- Sentiment analysis of news headlines using NLTK
- Integration of sentiment scores with stock price data
- Comparison of multiple machine learning and deep learning models
- Visualization of stock trends and prediction results

## Performance

The notebook compares the performance of different models in predicting stock price movements. Actual performance metrics are calculated and displayed within the notebook.

## Future Improvements

- Incorporate more advanced natural language processing techniques
- Experiment with ensemble methods for improved predictions
- Include additional financial indicators as features
- Implement real-time prediction capabilities using streaming data
