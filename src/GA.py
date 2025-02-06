import numpy as np
import random
import DataReader as dr

covMatrix = dr.covariance_matrix(dr.trainingData)


class GA:

    def __init__(self, populationSize=100, nGens=500, fitnessFunction='covariance', mutationRate=0.05,
                 crossoverProbability=0.7):
        self.populationSize = populationSize
        self.nGens = nGens
        self.fitnessFunction = fitnessFunction
        self.mutationRate = mutationRate
        self.crossoverProbability = crossoverProbability

    def validate_inputs(self):
        testReturns = []
        if self.nGens < 10:
            testReturns.append("Error: Number of generations must be >= 10")
        if self.populationSize < 1:
            testReturns.append("Error: Population size must be >= 1")
        if 0 >= self.crossoverProbability or self.crossoverProbability >= 1:
            testReturns.append("Error: Crossover probability must be on the interval [0, 1]")
        if 0 >= self.mutationRate or self.mutationRate >= 1:
            testReturns.append("Error: Mutation rate must be on the interval [0, 1]")

        if not testReturns:
            print("All conditions met, ready to run GA")
            return True
        else:
            for line in testReturns:
                print(line)
            return False

    def run_GA(self):
        # initialise population and weight vectors
        population, weightVectors = self.initialise_population()
        bestIndividuals = []
        bestIndividual = []
        for gen in range(self.nGens):
            if gen % round(self.nGens / 10) == 0 and gen != 0:
                print("reached gen", gen)
                bestIndividuals.append(bestIndividual)
            # evaluate the fitness scores
            fitnessScores = []
            for i in range(len(population)):
                score = self.calculate_fitness(weightVectors[i], self.fitnessFunction)
                fitnessScores.append(score)

            # begin selection
            selectionIDs = [i for i in range(len(population))]
            # crossover
            children = []
            for i in range(len(population)):
                parent1 = self.get_individual_from_id(self.selection(fitnessScores, selectionIDs), population)
                parent2 = self.get_individual_from_id(self.selection(fitnessScores, selectionIDs), population)
                c1, c2 = self.crossover(parent1, parent2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                children.append(c1)
                children.append(c2)
            # update population
            population = self.replace_population(population, weightVectors, (len(population) - len(children)), children)
            for i in range(len(population)):
                weightVectors[i] = dr.investment_weight_convert(population[i])
            gen += 1

            bestFitnessIdx = fitnessScores.index(max(fitnessScores))
            bestIndividual = dr.investment_weight_convert(population[bestFitnessIdx])

        return bestIndividual, bestIndividuals

    def initialise_population(self):
        population = []
        weightVectors = [[] for i in range(self.populationSize)]
        for i in range(self.populationSize):
            individual = []
            for j in range(len(dr.trainingStock)):
                # initialise genes as a random investment portfolio
                individual.append(round(random.uniform(0, 1), 3))
            population.append(individual)
            weightVectors[i] = dr.investment_weight_convert(individual)
        return population, weightVectors

    def get_individual_from_id(self, ID, population):
        for i in range(len(population)):
            if i == ID:
                return population[i]

    def selection(self, fitnessScores, IDs):
        # fitness scores, higher score means more chance of being selected
        minScore = min(fitnessScores)
        # to avoid negative values, shift scores by the abosolute of the lowest
        shiftedScores = [score + abs(minScore) for score in fitnessScores]
        sumFitness = sum(shiftedScores)
        probabilities = [shiftedScores[i]/sumFitness for i in range(len(fitnessScores))]
        chosen = np.random.choice(IDs, p=probabilities)
        return chosen

    def replace_population(self, population, weightVectors, numToReplace, children):
        # go through population, figure out fitness of each individual and compare it to the
        # child's fitness, if it is less than the child's, replace it with the child
        replacedIndices = []
        for child in children:
            cFitness = self.calculate_fitness(dr.investment_weight_convert(child), self.fitnessFunction)
            for i in range(len(population)):
                if numToReplace > 0 and i not in replacedIndices:
                    pFitness = self.calculate_fitness(weightVectors[i], self.fitnessFunction)
                    if pFitness < cFitness:
                        population[i] = child
                        replacedIndices.append(i)
                        numToReplace -= 1
        return population

    def crossover(self, p1, p2):
        # crossover probability
        rand = random.uniform(0, 1)
        if rand > self.crossoverProbability:
            return p1, p2
        # uniform crossover
        c1 = [0 for _ in range(len(p1))]
        c2 = [0 for _ in range(len(p1))]
        for i in range(len(p1)):
            if random.uniform(0, 1) < 0.5:
                c1[i] = p1[i]
                c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
        return c1, c2

    def calculate_fitness(self, weightVector, function):
        expectedReturn = dr.calculate_return_percentage(weightVector, dr.averageReturns)
        if function == "covariance":
            minRisk = dr.calculate_risk(weightVector, covMatrix)
            return expectedReturn / minRisk
        elif function == 'sharpe':
            return dr.sharpe_ratio(255, 0.01, [i * j for i, j in zip(weightVector, dr.averageReturns)])

    def mutate(self, c):
        # mutate child individual by altering one of its genes
        if random.uniform(0, 1) < self.mutationRate:
            c[random.randint(0, len(c) - 1)] = random.uniform(0, 1)
        return c
