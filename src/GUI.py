import math
import pygame as pg
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import DataReader as dr
from GA import GA
from ANN import ANN

pg.init()
pg.display.set_caption('Portfolio Optimiser')
WIDTH, HEIGHT = 800, 500

WHITE = 255, 255, 255

bgCol = (0, 43, 54)
screen = pg.display.set_mode((WIDTH, HEIGHT))
screen.fill(bgCol)
pg.display.flip()

# default GA parameters
possibleFs = ['covariance', 'sharpe']
fitnessF = possibleFs[0]
popsize = 100
gens = 500
mrate = 0.05
crossp = 0.7

# default ann parameters
activation = 'sigmoid'
possibleAs = ['sigmoid', 'softmax', "relu", 'linear', 'tanh']
epochs = 500
lossF = 'mae'
possibleLoss = ['mae', 'mse', 'binary_crossentropy', 'mean_square_error', 'mean_accuracy,error']


def write_to_terminal(text):
    posY = 0
    for line in text:
        textRect = pg.Rect(0, 0, 0, 32)
        textSurface = font.render(line, True, colour)
        screen.blit(textSurface, (0, posY))
        pg.draw.rect(screen, colour, textRect, 1)
        pg.display.flip()
        posY += 30
        time.sleep(0.05)


def reset_screen():
    inputRect = pg.Rect(0, HEIGHT - 50, 3 * WIDTH / 5, 38)
    screen.fill(bgCol)
    textSurface = font.render(command, True, colour)
    screen.blit(textSurface, (inputRect.x + 5, inputRect.y + 5))
    pg.draw.rect(screen, bgCol, inputRect, 1)
    pg.display.update(inputRect)


def help_command(command):
    # returns detailed help for a command of the form help <command>
    if command == "run":
        text = ["run:",
                "",
                "  * Runs the system, employing any set parameters",
                "  * Has no parameters",
                "  * Example: run"]
    elif command == "cdir":
        text = ["cdir:",
                "",
                "  * Changes directory where data is read",
                "  * One paramater <dir>, must be in the appropriate",
                "    directory format for your system",
                "  * CSV data must be the only folder(s) in directory",
                "  * Example: cdir /home/myFiles/data"]
    elif command == "dir":
        text = ["dir:",
                "",
                "  * Displays current data directory",
                "  * Has no parameters",
                "  * Example: dir",
                "  * Example output: /home/myFiles/data"]
    elif command == "csplitdate":
        text = ["csplitdate:",
                "",
                "  * Splits historical data into two train/test subsets",
                "    separated by the given date",
                "  * Training set is data up to the date, testing set is",
                "    data after the date",
                "  * Date must be in the form yyyy-mm-dd",
                "  * Example: csplitdate 2018-03-01"]
    elif command == "splitdate":
        text = ["splitdate:",
                "",
                "  * Displays the current Train/Test split date",
                "",
                "  * Example output: 2023-01-04",
                "  * Dates are of the form YYYY-MM-DD"]
    elif command == "ann":
        text = ["ANN:",
                "  Model used to make predictions",
                "",
                "  * \"epochs\"      - Number of times model reads dataset",
                "  * \"activation\"  - The function that activates a neuron",
                "                               E.g. ('relu', 'linear', 'sigmoid', 'tanh', 'softmax')",
                "  * \"lossF\"           -  Loss function to improve predictions",
                "                               E.g. ('mse', 'mae', 'poisson')"]
    elif command == "ga":
        text = ["GA:",
                "  Evolutionary algorithm to create a portfolio",
                "",
                "  * \"popsize\"      - Defines the population size",
                "  * \"gens\"           - Number of generations",
                "  * \"fitnessF\"     - Measures an individuals performance",
                "                              in the population E.g. ('covariance', 'sharpe')",
                "  * \"mrate\"          - Likelihood of an individual mutating a gene [0, 1]",
                "  * \"crossp\"             - Likelihood of two parent individuals reproducing [0, 1]"]
    elif command == "epochs":
        text = ["Epochs:",
                "",
                "  Number of times model reads dataset",
                "  * Higher epochs can lead to greater accuracy",
                "  * epochs <amount> to set number of epochs"]
    elif command == "activation" or command == "actv":
        text = ["Activation Function:",
                "  How the model activates a neuron",
                "",
                "  * Relu",
                "  * Linear",
                "  * Sigmoid",
                "  * Softmax",
                "  * Tanh"]
    elif command == "lossf" or command == "loss":
        text = ["Loss Function:",
                "  How the model measures its performance",
                "",
                "  * Mean Squared Error (mse)",
                "  * Mean Accuracy Error (mae)",
                "  * Binary Crossentropy (binary_crossentropy)",]
    elif command == "mrate" or command == "mutation":
        text = ["Mutation Rate:",
                "  Likelihood of an individual experiencing 'mutation'",
                "",
                "  * This project uses insertion mutation",
                "  * A random gene is selected in the array of values",
                "  * This random gene is then given a random value",
                "  * E.g.",
                "  * Before: [0.2, 0.3, 0.5, 0.1], randPos = 1, newGene = 0.6",
                "  * After: [0.2, 0.6, 0.5, 0.1]",
                "  * \"mrate <rate>\" to change mutation rate, on interval [0, 1]"]
    elif command == "crossp" or command == "crossover":
        text = ["Crossover probability:",
                "",
                "  Likelihood of two parent individuals reproducing'",
                "  * This project uses uniform crossover",
                "  * E.g. parent 1 = [0.5, 0.3, 0.2, 0.1]",
                "         parent 2 = [0.3, 0.8, 0.1, 0.9]",
                "",
                "         child 1 = [0.3, 0.3, 0.1, 0.9]",
                "         child 2 = [0.5, 0.8, 0.2, 0.1]"]
    elif command == "gens":
        text = ["Number of generation:",
                "  The number of generations in the algorithm",
                "",
                "  * Higher values => Higher search space exploration ",
                "  * \"gens <amount>\" to change number of generations"]
    elif command == "popsize" or command == "pop":
        text = ["Population size:",
                "  The number of individuals in the population",
                "",
                "  * Higher population => Higher search space exploration ",
                "  * \"popsize <size>\" to change population size"]
    elif command == "fitnessf" or command == "fitness":
        text = ["Fitness function:",
                "  Measures the likelihood of an individual being selected to reproduce",
                "",
                "  * Covariance, Sharpe",
                "  * \"fitnessF <function>\" to change fitness function"]
    else:
        text = ["Unknown command!",
                "Type \"help\" for list of commands"]
    return text


def parse_command(parsed):
    # parses command
    text = ''
    global epochs, activation, lossF, mrate, crossp, fitnessF, popsize, gens
    parsed = parsed.lower()
    if parsed == "help" or parsed == "help ":
        text = ["Commands:",
                "",
                "  * help <command>    - Detailed help for given command",
                "  * run                              - Runs system",
                "  * cdir <dir>                   - Changes directory where data is read",
                "  * dir                                - Returns current directory",
                "  * csplitdate <date>    - Split data into train/test by date",
                "  * splitdate                   - Returns current splitdate",
                "  * help ann                    - Detailed help for ann parameters",
                "  * help ga                      - Detailed help for ga parameters"
                ]
    elif "help " in parsed:
        text = help_command(parsed[5:])
    elif parsed == "run":
        text = run()
    elif parsed == "dir":
        text = ["Current directory:",
                "  " + dr.direc,
                "  * \"cdir <dir>\" to change directory"]
    elif "cdir " in parsed:
        text1 = ["Changing directory:",
                 "  " + dr.direc]
        dr.direc = dr.set_dir(parsed[5:])
        text2 = ["to:",
                 "  " + dr.direc]
        text = text1 + text2
    elif "csplitdate " in parsed:
        current = dr.splitDate
        if dr.set_split_date(parsed[11:]):
            text1 = ["Changing split date:",
                     "  " + current]
            text2 = ["to:",
                     "  " + dr.splitDate]
            text = text1 + text2
        else:
            write_to_terminal(["Error, date is not in valid format!",
                               "Format: YYYY-MM-DD",
                               "E.g: 2018-02-02"])
    elif parsed == "splitdate":
        text = ["Current split date:",
                "  " + dr.splitDate,
                "  * \"csplitdate <date>\" to change split date"]

    elif parsed == "epochs":
        text = ["Current epochs:",
                "  " + str(epochs),
                "  *\"epochs <amount>\" to change epochs"]
    elif "epochs " in parsed:
        try:
            es = int(parsed[6:])
            if es >= 1:
                text1 = ["Changing epochs:",
                         "  " + str(epochs)]
                epochs = es
                text2 = ["to:",
                         "  " + str(epochs)]
                text = text1 + text2
            else:
                text = ["Error: Number of epochs must be >= 1"]
        except ValueError as e:
            write_to_terminal([str(e),
                               "",
                               "Error: Epochs must be an integer value"])
    elif parsed == "activation":
        text = ["Current acivation:",
                "  " + activation,
                "  *\"activation <function>\" to change activation"]
    elif "activation " in parsed:
        a = parsed[11:]
        if a in possibleAs:
            text1 = ["Changing activation:",
                     "  " + activation]
            activation = a
            text2 = ["to:",
                     "  " + activation]
            text = text1 + text2
        else:
            text = ["",
                    "Error: Not an activation function in the possible choices",
                    "",
                    "Use \"help ann\" to view possible activation functions"]

    elif parsed == "lossf":
        text = ["Current loss function:",
                "  " + lossF,
                "  *\"lossF <function>\" to change loss function"]
    elif "lossf " in parsed:
        l = parsed[6:]
        if l in possibleLoss:
            text1 = ["Changing loss function:",
                     "  " + lossF]
            lossF = l
            text2 = ["to:",
                     "  " + lossF]
            text = text1 + text2
        else:
            text = ["",
                    "Error: Not a loss function in the possible choices",
                    "",
                    "Use \"help ann\" to view possible loss functions"]

    elif parsed == "popsize":
        text = ["Current population size:",
                "  " + str(popsize),
                "  *\"popsize <size>\" to change population size"]
    elif "popsize " in parsed:
        ps = int(parsed[8:])
        try:
            if ps >= 1:
                text1 = ["Changing population size:",
                         "  " + str(popsize)]
                popsize = ps
                text2 = ["to:",
                         "  " + str(popsize)]
                text = text1 + text2
            else:
                text = ["",
                        "Error: Population size must be >= 1"]
        except ValueError as e:
            text = ["",
                    str(e),
                    "",
                    "Error: Population size must be an integer >= 1"]

    elif parsed == "fitnessf":
        text = ["Current fitness function:",
                "  " + fitnessF,
                "  *\"fitnessF <function>\" to change fitness function"]
    elif "fitnessf " in parsed:
        f = parsed[9:]
        if f in possibleFs:
            text1 = ["Changing fitness function:",
                     "  " + fitnessF]
            fitnessF = parsed[9:]
            text2 = ["to:",
                     "  " + fitnessF]
            text = text1 + text2
        else:
            text = ["",
                    "Error: Not a possible choice of fitness function",
                    "",
                    "Use \"help ga\" to view possible fitness functions"]

    elif parsed == "mrate":
        text = ["Current mutation rate:",
                "  " + str(mrate),
                "  *\"mrate <rate>\" to change mutation rate"]
    elif "mrate " in parsed:
        try:
            mr = float(parsed[5:])
            if 0 <= mr <= 1:
                text1 = ["Changing mutation rate:",
                         "  " + str(mrate)]
                mrate = mr
                text2 = ["to:",
                         "  " + str(mrate)]
                text = text1 + text2
            else:
                text = (["",
                         "Error: Mutation rate must be on the interval [0,1]"])
        except ValueError as e:
            text = ([str(e),
                     "",
                     "Mutation rate must be a float value on the interval [0,1]",
                     "E.g: mrate 0.4"])

    elif parsed == "crossp":
        text = ["Current crossover probability:",
                "  " + str(crossp),
                "  *\"crossp <probability>\" to change crossover probability"]
    elif "crossp " in parsed:
        try:
            ncrossp = float(parsed[6:])
            if 0 <= ncrossp <= 1:
                text1 = ["Changing crossover probability:",
                         "  " + str(crossp)]
                crossp = ncrossp
                text2 = ["to:",
                         "  " + str(crossp)]
                text = text1 + text2
            else:
                text = ["",
                        "Error: Crossover Probability must be in the interval [0, 1]",
                        "E.g: crossp 0.5"]
        except ValueError as e:
            text = ([str(e),
                     "",
                     "Crossover probability must be a float on the interval [0, 1]"])

    elif parsed == "gens":
        text = ["Current number of generations:",
                "  " + str(gens),
                "  *\"gens <amount>\" to change number of generations"]
    elif "gens " in parsed:
        try:
            ngens = int(parsed[4:])
            if ngens < 10:
                text = ["",
                        "Error: Number of generations must be >= 10!"]
            else:
                text1 = ["Changing number of generations:",
                         "  " + str(gens)]
                gens = ngens
                text2 = ["to:",
                         "  " + str(ngens)]
                text = text1 + text2
        except ValueError as e:
            text = ([str(e),
                     "",
                     "Number of generations must be an integer!"])
    else:
        text = ["",
                "Unknown command!",
                "Type \"help\" for list of commands"]
    write_to_terminal(text)


def run():
    tp = 50
    write_to_terminal(dr.get_read_text())
    time.sleep(2)
    reset_screen()
    stockData = dr.stockData
    ga = GA(popsize, gens, fitnessF, mrate, crossp)
    try:
        text = ["Running GA..."]
        time.sleep(1)
        write_to_terminal(text)
        text += ["Successfully created population",
                 "Now carrying out selection, crossover and mutation ",
                 "Expected time: " + str(round(gens / 20)) + " seconds"]
        write_to_terminal(text)
        weightVec, best = ga.run_GA()
    except ValueError as e:
        text = ["Genetic algorithm error occurred",
                "  * " + str(e),
                "",
                "Possible reason:",
                "  * Split date is likely too close to one side of date range",
                "  * Try using \"csplitdate <date>\" to adjust split date"]
        return text
    text += ["",
             "The optimal portfolio found by the system is:",
             str(weightVec)]
    time.sleep(1.5)
    text += [" Here were the best individuals",
             " from every " + str(round(gens/10)) + " generations"]
    write_to_terminal(text)
    i = 1
    for ind in best:
        if len(text) > 14:
            reset_screen()
            text = []
        text += [str(ind) + " from gen " + str(math.floor(i * (gens/10)))]
        i += 1
        time.sleep(0.5)
        write_to_terminal(text)
    text += ["and",
             str(weightVec) + " from gen " + str(gens)]

    text += ["Running ANN...",
             ""]
    write_to_terminal(text)
    ann = ANN(activation, None, epochs, lossF)
    # risk = dr.calculate_risk(weightVec, dr.covariance_matrix(stockData))
    risk = dr.ann_risk(weightVec)
    frame = ann.create_ann_data(risk, tp)
    scalar = StandardScaler()
    x_train, x_test, y_train, y_test = ann.split_data(frame)

    x_train = scalar.fit_transform(x_train)
    x_test = scalar.fit_transform(x_test)
    ann.compile_model(x_train, y_train)
    testPredict = ann.run_model(x_train, y_train, x_test)
    testPredict = [[row[i] for row in testPredict] for i in range(len(testPredict[0]))]

    plot_results(testPredict, y_test)
    text += ["",
             "Successfully ran ANN!",
             "Now saving plots to \"plots\" folder",
             "",
             "The best portfolio was:",
             str(weightVec)]
    write_to_terminal(text)
    return ''


def read_terminal(event, command):
    if event.key == pg.K_BACKSPACE:
        if command[-2] == ">":
            return '>_'
        command = command[:-2] + '_'
    elif event.key == pg.K_RETURN:
        parse_command(command[1:-1])
        command = '>_'
    else:
        command = command[:-1]
        command += event.unicode + '_'
    return command


def plot_results(results, y_test):
    dates = dr.stockData[0]['Date']
    datesY = dates.iloc[y_test.index[0]:]
    #
    # print(y_test)
    # print(results)

    for i in range(len(dr.trainingStock)):
        plt.plot(datesY, results[i])
        plt.plot(datesY, y_test['Price Change_' + str(i)])
        plt.legend(['Test Predictions', 'Actual'])
        plt.title(dr.trainingStock[i])
        plt.ylim(-0.06, 0.06)
        plt.savefig("plots/" + dr.trainingStock[i] + ".png")
        plt.close()


# main
font = pg.font.Font("src/Exo-Regular.otf", 24)
colour = WHITE
done = False
command = '>_'
write_to_terminal(["",
                   "Welcome!",
                   "",
                   "Type \"help\" for a list of commands",
                   "",
                   "Type \"run\" to run the system"])
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        elif event.type == pg.KEYDOWN:
            command = read_terminal(event, command)

    # Render the text input
    reset_screen()
