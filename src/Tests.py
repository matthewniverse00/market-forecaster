from GA import GA
import DataReader as dr
import random
from ANN import ANN, evaluate_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

covMatrix = dr.covariance_matrix(dr.trainingData)
covMatrixTest = dr.covariance_matrix(dr.testData)


def display_results_GA():
    ga = GA(100, 1000, 'covariance', 0.5, 0.7)
    weight, pop = ga.run_GA()
    balancedPort = [round(1 / len(dr.trainingStock), 4) for stock in dr.trainingStock]
    randomPort = [random.uniform(0, 1) for stock in dr.trainingStock]
    print()
    print("########## Covariance based fitness ##########")
    print("########## Training Data ##########")

    print("The best individual was")
    print(weight)
    ret = round(dr.calculate_return_percentage(weight, dr.averageReturns), 4)
    risk = round(dr.calculate_risk(weight, covMatrix), 4)
    print("with stats")
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    print("Balanced port stats")
    print(balancedPort)
    ret = round(dr.calculate_return_percentage(balancedPort, dr.averageReturns), 4)
    risk = round(dr.calculate_risk(balancedPort, covMatrix), 4)
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    randWeight = dr.investment_weight_convert(randomPort)
    print("Random port stats")
    print(randWeight)
    ret = round(dr.calculate_return_percentage(randWeight, dr.averageReturns), 4)
    risk = round(dr.calculate_risk(randWeight, covMatrix), 4)
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    print("########## Test Data ##########")
    print("Evolved portfolio returns and risk")
    print(weight)
    ret = round(dr.calculate_return_percentage(weight, dr.averageReturnsTest), 4)
    risk = round(dr.calculate_risk(weight, covMatrixTest), 4)
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    print("Balanced portfolio returns and risk")
    print(balancedPort)
    ret = round(dr.calculate_return_percentage(balancedPort, dr.averageReturnsTest), 4)
    risk = round(dr.calculate_risk(balancedPort, covMatrixTest), 4)
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    print("Random portfolio returns and risk")
    print(randWeight)
    ret = round(dr.calculate_return_percentage(randWeight, dr.averageReturnsTest), 4)
    risk = round(dr.calculate_risk(randWeight, covMatrixTest), 4)
    print("Return:", ret, " Risk:", risk)
    print("Ratio:", round(ret/risk, 4))
    print()
    # ga = GA(100, 1000, 'sharpe', 0.5, 0.7)
    # weight, pop = ga.run_GA()
    # print()
    # print("########## Sharpe ratio based fitness ##########")
    # print("########## Training data ##########")
    # print()
    # print("Evolved portfolio Returns and Sharpe Ratio")
    # print(weight)
    # print(dr.calculate_return_percentage(weight, dr.averageReturns))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(weight, dr.averageReturns)]))
    # print()
    # print("Balanced portfolio Returns and Sharpe Ratio")
    # print(balancedPort)
    # print(dr.calculate_return_percentage(balancedPort, dr.averageReturns))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(balancedPort, dr.averageReturns)]))
    # print()
    # print("Random portfolio Returns and Sharpe Ratio")
    # print(randWeight)
    # print(dr.calculate_return_percentage(randWeight, dr.averageReturns))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(randWeight, dr.averageReturns)]))
    # print()
    # print("########## Test Data ##########")
    # print("Evolved")
    # print(weight)
    # print(dr.calculate_return_percentage(weight, dr.averageReturnsTest))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(weight, dr.averageReturnsTest)]))
    # print()
    # print("Balanced")
    # print(balancedPort)
    # print(dr.calculate_return_percentage(balancedPort, dr.averageReturnsTest))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(balancedPort, dr.averageReturnsTest)]))
    # print()
    # print("Random")
    # print(randWeight)
    # print(dr.calculate_return_percentage(randWeight, dr.averageReturnsTest))
    # print(dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(randWeight, dr.averageReturnsTest)]))
    # print()

    intP = str(input("View other statistics?: "))
    if intP.lower() == 'y':
        print("Other statistics")
        for indv in pop:
            weight = dr.investment_weight_convert(indv)
            print("Return:", dr.calculate_return_percentage(weight, dr.averageReturns), " Risk:",
                  dr.calculate_risk(weight, covMatrix))
            print()
    else:
        exit()


def gen_unit_tests_ga():
    ga = GA(populationSize=10)
    print("Genetic Algorithm Generation Unit tests")
    ga.nGens = -1
    ga.populationSize = 10
    ga.mutationRate = 0.0
    ga.crossoverProbability = -0.1
    print_ga(ga)

    ga.nGens = 1
    ga.populationSize = 10
    ga.mutationRate = 0.0
    ga.crossoverProbability = 0.0
    print_ga(ga)

    ga.nGens = 1000
    ga.populationSize = 100
    ga.mutationRate = 0.05
    ga.crossoverProbability = 0.7
    print_ga(ga)

    ga.nGens = 10000
    ga.populationSize = 1000
    ga.mutationRate = 1.0
    ga.crossoverProbability = 1.1
    print_ga(ga)


def print_ga(ga):
    print()
    print("Testing Genetic Algorithm with: ", ga.populationSize, " population size")
    print("                                ", ga.nGens, " generations")
    print("                                ", ga.mutationRate, " mutation rate")
    print("                                ", ga.crossoverProbability, " crossover probability")
    print(ga.validate_inputs())
    print()


def evaluate_ann_parameters():
    # known good weight portfolio
    weight = [0.018, 0.531, 0.384, 0.066]
    # set up activation + loss functions to iterate over
    activations = ['relu', 'linear', 'softmax', 'sigmoid', 'tanh']
    lossFs = ['mse', 'mae', 'binary_crossentropy']
    # instantiate ANN model, split training and testing data
    ann = ANN()
    risk = dr.ann_risk(weight)
    frame = ann.create_ann_data(risk, 50)
    x_train, x_test, y_train, y_test = ann.split_data(frame)

    print("Running (200 epochs)")
    for loss in lossFs:
        for f in activations:
            # create new ann instance with current loss and activation function
            ann = ANN(f, None, 200, loss)
            accuracies = []
            # loop 3 times for average
            for j in range(3):
                pred = ann.run_model(x_train, y_train, x_test)
                a = evaluate_model(y_test, pred, 3)
                accuracies.append(a)
            stockAccuracies = [0 for i in range(len(dr.trainingStock))]
            # take average of losses for 3 runs
            print(ann.activation, " paired with ", ann.lossF)
            for i in range(len(accuracies[0])):
                acc = sum([accuracies[j][i] for j in range(len(accuracies))])
                stockAccuracies[i] = round(acc / 3, 4)
            print(stockAccuracies)
            print()


def evaluate_ann_portfolio():
    # known good weight portfolio, rounded up a little
    weight = [0.21, 0.3, 0.29, 0.2]
    ann1 = ANN('sigmoid', None, 100, 'binary_crossentropy')
    risk = dr.calculate_risk(weight, covMatrixTest)
    frame = ann1.create_ann_data(risk, 50)
    x_train, x_test, y_train, y_test = ann1.split_data(frame)

    accuracies = []
    for i in range(4):
        pred = ann1.run_model(x_train, y_train, x_test)
        a = evaluate_model(y_test, pred, 4)
        accuracies.append(a)

    stockAccuracies = [0 for i in range(len(dr.trainingStock))]
    for i in range(len(accuracies[0])):
        acc = sum([accuracies[j][i] for j in range(len(accuracies))])
        stockAccuracies[i] = round(acc / 3, 4)

    print("Activ:,", ann1.activation, "LossF:", ann1.lossF)
    print("Results with considering evolved portfolio")
    print(stockAccuracies)
