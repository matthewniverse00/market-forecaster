import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras.utils.io_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import DataReader as dr


keras.utils.io_utils.disable_interactive_logging()


def evaluate_model(y, pred, size):
    # we want to see how often the model accurately predicts price momentum
    direcAccs = []
    for i in range(len(pred[0])):
        # store percentage change in returns from today
        yD = [y.iloc[i][j] for j in range(len(y.iloc[0]))]
        yPD = pred[i]
        # take the difference between consecutive values e.g. yD[i], yD[i+1]
        # then extract their sign
        yDirec = np.sign((np.diff(yD)))
        yPredDirec = np.sign(np.diff(yPD))
        # count how many of the predicted signs match the actual
        direcs = np.sum(yDirec == yPredDirec)
        # take accuracy
        direcAcc = round(direcs / size, 5)
        # append to list of stock asset prediction accuracies
        direcAccs.append(direcAcc)
    return direcAccs


class ANN:

    def __init__(self, activation='sigmoid', model=None, epochs=300, lossF='mse'):
        self.activation = activation
        self.model = model
        self.epochs = epochs
        self.lossF = lossF

    def pred(self, data):
        return self.model.predict(data)

    def run_model(self, x, y, xT):
        self.compile_model(x, y)
        # fit training data in, technical indicators and their actual output
        # 20% goes to validating model
        preds = self.model.predict(xT)
        return preds

    def compile_model(self, x, y):
        # creating model, 192 neurons
        length = len(dr.trainingStock)
        self.model = Sequential()
        self.model.add(Dense(length * 5, activation=self.activation, input_dim=x.shape[1]))
        self.model.add(Dense(length * 5, activation=self.activation))
        self.model.add(Dense(length * 3, activation=self.activation))
        self.model.add(Dense(length * 2, activation=self.activation))
        # model has n outputs for n stock assets
        self.model.add(Dense(length))
        self.model.compile(loss=self.lossF, optimizer='adam')
        self.model.fit(x, y, epochs=self.epochs, validation_split=0.1)

    def create_ann_data(self, portRisk, tp):
        # create dataframe of technical indicators and the return/risk ratio they associate with
        stockData = dr.stockData
        merged = pd.DataFrame()
        for i in range(len(stockData)):
            stock = dr.calculate_technical_indicators(stockData[i], portRisk, i, tp).dropna()
            merged = pd.concat(
                [merged, stock['ROC_' + str(i)], stock['RSI_' + str(i)], stock['Return/Risk_' + str(i)],
                 stock['Price Change_' + str(i)]], axis=1)
        return merged

    def split_data(self, frame):
        # split data into train and test sets to evaluate model
        dataSplitX = [
            ['ROC_' + str(i), 'RSI_' + str(i),
             'Return/Risk_' + str(i)] for
            i in
            range(len(dr.trainingStock))]
        dataSplitCatX = []
        dataSplitY = [['Price Change_' + str(i)] for i in range(len(dr.trainingStock))]
        dataSplitCatY = []
        for array in dataSplitX:
            dataSplitCatX += array
        for array in dataSplitY:
            dataSplitCatY += array
        X = frame[dataSplitCatX]
        Y = frame[dataSplitCatY]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
        return x_train, x_test, y_train, y_test
