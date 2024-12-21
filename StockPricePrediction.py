##################### STOCK PRICE PREDICTION ####################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SPPDataSet = pd.read_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/StockPricePredictionDataset.csv')
print(SPPDataSet.head())
 
################### Exploratory Data Analysis ######################

# Check the shape of the dataset
print("Dimension Of the Dataset is : ",SPPDataSet.shape)

# Check column names and data types
print("Column Names and its data types : ")
print(SPPDataSet.info())

# Checking for missing values
print("Total Missing Values Present : ")
print(SPPDataSet.isnull().sum())


# Plotting Adj Close
plt.figure(figsize=(20, 8))
plt.plot(SPPDataSet['Close'])
plt.title('Apple Stocks Trend')
plt.show()

# checking for outliers
sns.violinplot(data=SPPDataSet[['Open', 'High', 'Low','Close','Adj Close']])
plt.title('violin plot of all the Features')
plt.xlabel('Features')
plt.show()

statisticalMeasures=SPPDataSet.describe()
print("Statistical Measures : ")
print(statisticalMeasures)
statisticalMeasures.to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/statisticalMeasures.csv',
index=False)

# Plotting Means and Variances of the features

means = pd.DataFrame(SPPDataSet[['Open', 'High', 'Low','Close','Adj Close']]).mean()
plt.bar(means.index, means.values)
plt.xlabel('Features')
plt.ylabel('Mean')
plt.title('Mean of Features')
plt.show()

variances = pd.DataFrame(SPPDataSet[['Open', 'High', 'Low','Close','Adj Close']]).var()
plt.bar(variances.index, variances.values)
plt.xlabel('Features')
plt.ylabel('variances')
plt.title('Variances of Features')
plt.show()


# Compute correlation matrix
corr_matrix = SPPDataSet[['Open', 'High', 'Low','Close','Adj Close','Volume']].corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

#################### Data Preprocessing ############################

# Convert 'Date' column to datetime format and set it as index
SPPDataSet['Date'] = pd.to_datetime(SPPDataSet['Date'])
SPPDataSet.set_index('Date', inplace=True)

# Extract the 'Close' column for further analysis
closing_prices = SPPDataSet.filter(['Close']).values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)
pd.DataFrame(scaled_prices).to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/closingPrices_Scaled.csv')

#################### Modelling the Data ############################


training_data = scaled_prices[:int(len(scaled_prices) * 0.8), :]
trainData_input, trainData_output = [], []

for i in range(60, len(training_data)):
    trainData_input.append(scaled_prices[i-60:i, 0])  # Last 'time_step' values
    trainData_output.append(scaled_prices[i, 0])  # Next day's close price


trainData_input, trainData_output = np.array(trainData_input), np.array(trainData_output)
pd.DataFrame(trainData_input).to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/TrainData_Input.csv')
pd.DataFrame(trainData_output).to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/TrainData_Output.csv')

# Reshape the data for LSTM
trainData_input = np.reshape(trainData_input, (trainData_input.shape[0], trainData_input.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(trainData_input.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

################################ Modeling with 20 Epochs ##########################

# Fit the model to the training data
model.fit(trainData_input, trainData_output, epochs=20, batch_size=32)

# Create the testing dataset
testing_data = scaled_prices[int(len(scaled_prices) * 0.8) - 60:, :]
testData_input, testData_ActualOutput = [], closing_prices[int(len(closing_prices) * 0.8):, :]

# Prepare testing data
for i in range(60, len(testing_data)):
  testData_input.append(testing_data[i-60:i, 0])
  
pd.DataFrame(testData_input).to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/TestData_Input.csv')
pd.DataFrame(testData_ActualOutput).to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/TestData_ActualOutput.csv')

testData_input = np.array(testData_input)
testData_input = np.reshape(testData_input, (testData_input.shape[0], testData_input.shape[1], 1))

# Get the models predicted price values
predictedValues1 = model.predict(testData_input)
predictedValues1 = scaler.inverse_transform(predictedValues1)

training_set = SPPDataSet.iloc[:int(len(closing_prices) * 0.8)]
test_results1 = SPPDataSet.iloc[int(len(closing_prices) * 0.8):]
test_results1.loc[:, 'Predictions'] = predictedValues1.copy()
test_results1.to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/ActualVsPredictedPricesWith20Epochs.csv')

# Visualize the predicted prices compared to actual prices
plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction using LSTM With 20 Epochs')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(training_set['Close'], label='Training Data')
plt.plot(test_results1[['Close', 'Predictions']], label=['Actual Prices', 'Predicted Prices'])
plt.legend(loc='lower right')
plt.show()

# Model Evaluation Metrics
mse_lstm = mean_squared_error(test_results1['Close'], test_results1['Predictions'])
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(test_results1['Close'], test_results1['Predictions'])
mape_lstm = np.mean(np.abs((test_results1['Close'] - test_results1['Predictions']) / test_results1['Close'])) * 100
r2_lstm = r2_score(test_results1['Close'], test_results1['Predictions'])

print('\nLSTM Model Evaluation With 20 Epochs:')
print(f'MSE: {mse_lstm:.2f}')
print(f'RMSE: {rmse_lstm:.2f}')
print(f'MAE: {mae_lstm:.2f}')
print(f'MAPE: {mape_lstm:.2f}%')
print(f'R2 Score: {r2_lstm:.2f}')

################################ Modeling with 25 Epochs ##########################

# Fit the model to the training data
model.fit(trainData_input, trainData_output, epochs=25, batch_size=32)

# Get the models predicted price values
predictions2 = model.predict(testData_input)
predictions2 = scaler.inverse_transform(predictions2)

test_results2 = SPPDataSet.iloc[int(len(closing_prices) * 0.8):]
test_results2.loc[:, 'Predictions'] = predictions2.copy()
test_results2.to_csv('C:/Users/Naveena/Documents/UNCG/Fall 2024/Assignments/Big Data/StockPricePredictionUsingLSTM/Data/Spreadsheets/ActualVsPredictedPricesWith25Epochs.csv')

# Visualize the predicted prices compared to actual prices
plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction using LSTM with 25 Epochs')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(training_set['Close'], label='Training Data')
plt.plot(test_results2[['Close', 'Predictions']], label=['Actual Prices', 'Predicted Prices'])
plt.legend(loc='lower right')
plt.show()

# Model Evaluation Metrics
mse_lstm = mean_squared_error(test_results2['Close'], test_results2['Predictions'])
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(test_results2['Close'], test_results2['Predictions'])
mape_lstm = np.mean(np.abs((test_results2['Close'] - test_results2['Predictions']) / test_results2['Close'])) * 100
r2_lstm = r2_score(test_results2['Close'], test_results2['Predictions'])

print('\nLSTM Model Evaluation With 25 Epochs:')
print(f'MSE: {mse_lstm:.2f}')
print(f'RMSE: {rmse_lstm:.2f}')
print(f'MAE: {mae_lstm:.2f}')
print(f'MAPE: {mape_lstm:.2f}%')
print(f'R2 Score: {r2_lstm:.2f}')