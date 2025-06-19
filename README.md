# Stock_Price_Prediction_Using_LSTM



Executive Summary

This project aims to predict the stock price of Tata Global Beverages (now Tata Consumer Products) using a deep learning technique called Long Short-Term Memory (LSTM). We utilize historical stock market data to train an LSTM model capable of learning temporal patterns. The final result is a predictive model that shows the ability to forecast short-term stock prices with reasonable accuracy. The project demonstrates the power of recurrent neural networks for time series forecasting and sets the foundation for further real-time deployment.


Introduction

Problem Statement

Stock price prediction is a challenging task due to the non-linear, noisy, and time-dependent nature of financial markets. Investors and traders seek predictive systems that can assist in making informed decisions. Traditional statistical models struggle to capture long-term dependencies in time series data.


Objective

The goal of this project is to build a predictive model using LSTM to forecast future stock prices of Tata Global based on historical opening prices. The model is expected to capture hidden trends and temporal patterns in the data for more accurate forecasting.



Data Exploration and Preprocessing

Data Source
•	Training Data: NSE-TATAGLOBAL.csv
•	Test Data: tatatest.csv


Description
•	Features: Date, Open, High, Low, Last, Close, Total Trade Quantity, Turnover
•	For this project, only the ‘Open’ price is used to build the model.


Preprocessing Steps
1.	Loaded training and test datasets using pandas.
2.	Extracted the 'Open' price column for modeling.
3.	Scaled the values to the range [0,1] using MinMaxScaler.
4.	Created time series sequences of 60 timesteps to predict the next (61st) value.
5.	Reshaped data into 3D structure as required by LSTM: (samples, timesteps, features).



Methodology

Model Used

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) suitable for time series prediction due to its ability to learn long-term dependencies.

Model Architecture
•	4 LSTM layers with 50 units each
•	Dropout layers (rate = 0.2) to prevent overfitting
•	Dense layer with 1 output unit

python
CopyEdit
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(Dropout(0.2))
...
model.add(Dense(1))


Training Configuration
•	Optimizer: Adam
•	Loss Function: Mean Squared Error (MSE)
•	Epochs: 100
•	Batch Size: 32


Model Evaluation
Loss Metric
•	Training Loss: Tracked using Mean Squared Error (lower is better)
•	Visual inspection of predicted vs real values was used for evaluation (since no formal accuracy metric is defined for regression)


Visualization
python
CopyEdit
plt.plot(real_stock_price, label='Actual')
plt.plot(predicted_stock_price, label='Predicted')
•	The model was able to follow the general trend of the stock prices with reasonable accuracy.



Application and Future Work

Real-world Application
•	Investors can use this model for short-term predictions.
•	Can be integrated into automated trading systems or financial dashboards.

Future Improvements
•	Include more features: High, Low, Volume, Technical Indicators (e.g., RSI, MACD).
•	Use live data from financial APIs (e.g., Alpha Vantage, Yahoo Finance).
•	Apply hyperparameter tuning with GridSearchCV or Keras Tuner.
•	Add evaluation metrics like RMSE, MAE.
•	Deploy using Streamlit, Dash, or Flask as a web application.


Conclusion
This project demonstrated the use of deep learning, particularly LSTM networks, for time series forecasting. Despite using only one feature ("Open" price), the model achieved visually acceptable results. With further improvements, such models can play an essential role in real-time stock trading and financial forecasting.


References
1.	Derrick Mwiti GitHub Repo: Stock Price Dataset
2.	Keras Documentation
3.	Scikit-learn Documentation
4.	MinMaxScaler - sklearn
5.	Investopedia - Understanding LSTMs and Stock Prediction


Acknowledgements
•	Special thanks to Derrick Mwiti for the dataset.
•	Thanks to the creators of open-source tools: Keras, TensorFlow, Pandas, and Matplotlib.
•	This project was inspired by the growing interest in using AI for financial applications.
