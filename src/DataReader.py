import statistics
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

# list of stocks to train GA on
# to keep things simple, this iteration involves the use of 4 chosen stocks
# apple, google, microsoft and sony
# others can be added by simply uploading the data to the "data" folder

direc = 'data/'
trainingStock = []
for csv in os.listdir(direc):
    if csv.endswith('.csv'):
        name, _ = os.path.splitext(csv)
        trainingStock.append(name)
# read stock price data from trainingStock list, store in list
text = []
stockData = []
try:
    stockData = [pd.read_csv(direc + f'{stockName}.csv') for stockName in trainingStock]
    print("Successfully read files")
except FileNotFoundError as e:
    print("Invalid directory, use \"dir\" to check what directory current is")
averageReturns = []
averageReturnsTest = []

# split train and test data, 4 years for training and 1 for testing
splitDate = '2019-09-02'
text.append("Current split date is: " + splitDate)

trainingData = []
testData = []


def get_read_text():
    return text


# store each security in a list, evaluate the average closing value in a 12-month period and the days return
for x, stockName in enumerate(stockData):
    # update data to add these new columns
    stockName['Date'] = pd.to_datetime(stockName['Date'])
    stockName['Close PCT'] = stockName['Adj Close'].pct_change()
    stockTrainingData = stockName.loc[stockName['Date'] <= splitDate]
    stockTestData = stockName.loc[stockName['Date'] > splitDate]
    averageReturns.append(stockTrainingData['Close PCT'].iloc[-1])
    averageReturnsTest.append(stockTestData['Close PCT'].iloc[-1])
    trainingData.append(stockTrainingData)
    testData.append(stockTestData)

text.append("GA train and test split successful")


def covariance_matrix(data):
    closeData = training_close_data(data)
    mergedData = pd.concat(closeData, axis=1)
    return mergedData.corr()


def set_dir(newDir):
    return newDir


def get_gens(gens):
    return gens


def set_split_date(newDate):
    global splitDate
    newDateC = newDate[:]
    try:
        splitDate = newDateC.split('-')
        if len(splitDate[0]) == 4 and len(splitDate[1]) + len(splitDate[2]) == 4:
            splitDate = newDate
            return True
    except IndexError:
        return False


def training_close_data(data):
    closeData = []
    for i in range(len(data)):
        closeData.append(data[i]['Close'])
        closeData[i] = closeData[i].rename(trainingStock[i])
    return closeData


def calculate_return_percentage(weightVector, returns):
    return np.dot(returns, weightVector)


def investment_weight_convert(individual):
    # weights represent what % of the investment goes to each security (stock asset)
    totalSecurities = len(trainingStock)
    weightVector = []
    for i in range(totalSecurities):
        weightVector.append(round(float(individual[i] / sum(individual)), 3))
    return weightVector


def ann_risk(weight):
    weight = np.array(weight)
    covMatrix = np.array(covariance_matrix(trainingData))
    var = np.dot(weight.T, np.dot(covMatrix, weight))
    return np.sqrt(var)


def sharpe_ratio(n, rf, returns):
    mean = (statistics.mean(returns) * n) - rf
    sigma = (statistics.stdev(returns)) * np.sqrt(n)
    # in order to keep consistency among fitness functions, sharpe ratio inverse will be used
    return sigma / mean


def calculate_technical_indicators(data, risk, i, tp):
    # data['SMA_' + str(i)] = ta.sma(data['Adj Close'], timeperiod=round(tp / 10))
    # data['LMA_' + str(i)] = ta.sma(data['Adj Close'], timeperiod=tp)
    # data['EMA_' + str(i)] = ta.ema(data['Adj Close'], timeperiod=round(tp / 10))
    # data['LEMA_' + str(i)] = ta.ema(data['Adj Close'], timeperiod=tp)
    data['ROC_' + str(i)] = ta.roc(data['Adj Close'], timeperiod=round(tp / 5))
    data['RSI_' + str(i)] = ta.rsi(data['Adj Close'], timeperiod=round(tp / 10))
    data['Return/Risk_' + str(i)] = round((data['Adj Close'].pct_change(periods=2) / risk), 3)
    data['Price Change_' + str(i)] = data['Adj Close'].pct_change()
    return data


def calculate_risk(weightVector, covMatrix):
    minRisk = 0.0
    for i in range(len(weightVector)):
        covariantSecurity = 0.0
        for j in range(len(weightVector)):
            covariantSecurity += weightVector[j] * covMatrix[trainingStock[i]][trainingStock[j]]
        minRisk += weightVector[i] * covariantSecurity
    return minRisk
