CrowdWisdom Sentiment & Price Prediction

Project Overview

This project explores whether social media sentiment and market data can help forecast NVIDIA (NVDA) stock prices over a five-trading-day horizon. It combines historical stock prices, sentiment analysis of X posts, and recent financial news to develop and evaluate a quantitative forecasting workflow.

The project was developed as part of the CrowdWisdomTrading Quantitative Data Scientist Internship Assignment.

Objectives

The main objectives are to collect historical NVDA stock price data and social media posts, analyse sentiment in X posts and financial news, engineer price-based and sentiment-based features, forecast closing prices for the next five trading days, evaluate model performance using time-series validation, and assess a prediction-based trading strategy using risk-adjusted performance metrics.

Technologies Used

The project uses Python and Google Colab for development, yfinance for historical stock price data, Apify for X post collection, Tavily for financial news retrieval, VADER Sentiment for text analysis, Pandas and NumPy for data processing and feature engineering, Scikit-learn for Random Forest regression and time-series validation, and Matplotlib for data visualisation.

Methodology

Data Collection

Historical NVDA price data was collected using yfinance. X posts mentioning NVDA were collected through Apify, and recent financial news was retrieved using Tavily.

Sentiment Analysis

VADER was used to calculate compound sentiment scores and classify text as positive, neutral, or negative. Daily sentiment features included average sentiment, median sentiment, tweet counts, and news sentiment measures.

Feature Engineering

Price-based features included daily and five-day returns, rolling volatility, volume changes, and lagged closing prices. Sentiment features were combined with market data to investigate whether they improved forecasting performance. Missing sentiment observations were distinguished from observed sentiment using availability indicators.

Price Forecasting

A Random Forest regression model was used to forecast NVDA closing prices over a five-trading-day horizon. The latest observed closing price was used as the current-day reference rather than treating it as a future prediction.

Model Evaluation

Walk-forward validation using TimeSeriesSplit was used to evaluate out-of-sample predictions. Mean Absolute Error (MAE) was compared with a persistence baseline, which assumes that the next closing price will equal the current closing price. An exploratory prediction-based trading strategy was also evaluated using cumulative returns, Sharpe ratio, Sortino ratio, and maximum drawdown.

Forecast Results

The latest available closing price in the collected dataset was $214.13 on October 16, 2026. The model generated the following estimates for the next five trading sessions: $213.93 for October 19, $213.31 for October 20, $215.24 for October 21, $217.81 for October 22, and $222.61 for October 23, 2026.

The October 16 closing price represents the observed current-day value (n+0), while the subsequent values are model-generated forecasts (n+1 to n+5). These estimates are not guaranteed future prices or investment recommendations.

Model Evaluation

For one-day-ahead forecasting, the price-only Random Forest model achieved a Mean Absolute Error (MAE) of 4.7573, while the price-plus-sentiment model achieved an MAE of 4.7515. The persistence baseline achieved a lower MAE of 4.0350.

Although adding sentiment features resulted in a very small improvement over the price-only model, both Random Forest models performed worse than the persistence baseline. This indicates that the current modelling approach did not demonstrate a reliable predictive advantage.

Trading Strategy Evaluation

An exploratory strategy took a long position when the predicted next closing price exceeded the current closing price. The walk-forward simulation covered 100 out-of-sample observations, with long positions taken on 48 days. The recorded cumulative strategy return was 56.13%, with a Sharpe ratio of 3.97, a Sortino ratio of 8.17, and a maximum drawdown of -8.14%.

These figures represent historical simulation outputs rather than evidence of live profitability. They should be interpreted cautiously because the evaluation is limited and does not account for transaction costs, slippage, taxes, or other real-world execution constraints.

Key Limitations

Although the price dataset covers approximately six months, the collected X posts were concentrated on only a few dates. Consequently, social sentiment had limited daily coverage, restricting conclusions about its predictive value.

Missing sentiment observations were filled with zero for modelling, while availability indicators were retained to distinguish missing data from observed sentiment. News sentiment was based on text scoring and may not fully capture financial context, sarcasm, or the actual market impact of an article.

The Random Forest models did not outperform the persistence baseline in the reported one-day-ahead MAE comparison. The trading simulation also excludes transaction costs and slippage, so its results should not be interpreted as a realistic estimate of achievable returns.

Repository Contents

The repository contains Copy_of_CrowdWisdom_Sentiment_Price_Prediction.ipynb, the main notebook covering data collection, preprocessing, sentiment analysis, feature engineering, forecasting, evaluation, and visualisations. It also includes nvda_forecast.csv, containing the forecast output, and nvda_walk_forward_results.csv, containing the out-of-sample walk-forward evaluation results. The README.md file provides the project documentation.
