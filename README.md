# wti-crude-oil-price-prediction
A Python machine learning project for predicting WTI Crude Oil prices using ridge regression


This project uses Python and a variety of robust libraries to achieve a WTI crude oil price prediction ridge regression model.
The model leverages historical price data and various statistical techniques to forecast oil prices. The code in this project includes all the necessary features to process and visualize the data to make the predictions and includes the following features: 

* Historical Data Retrieval: Using Yahoo Finance, this project retrieves WTI Crude Oil historical data that includes the opening, high, low,
and closing prices and traded volume.

* Data preprocessing: This program uses StandardScaler to scale the data for better performance.

* Model Training and Prediction: After trying a variety of models, Ridge regression produced the best results.
  The model is also evaluated using MSE, R-Squared and MAPE.

* Prediction Accuracy: Calculates and displays the prediction accuracy percentage based on the MAPE, which measures how well the model   performs in forecasting prices

**Libraries Used**

* matplotlib -- used to visualize the data with graphs
* pandas -- to handle and manipulate the financial data
* numpy -- for computations
* yfinance -- used to retrieve financial data from Yahoo Finance
* sklearn -- machine learning algorithms and processing

**How to install this program** 

1. Makes sure you have Python installed
2. Install required libraries <pip install matplotlib pandas numpy yfinance scikit-learn>
3. Clone the repository <git clone https://github.com/robertonava08/wti-crude-oil-price-prediction.git>
4. Go to project directory <cd wti-crude-oil-price-prediction>
5. Run project <python Crude_Oil_Predictor.py>

