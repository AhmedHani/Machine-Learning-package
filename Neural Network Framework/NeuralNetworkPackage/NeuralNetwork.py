__author__ = 'ahani'

import numpy as np
import copy as cp
from Utilities import *


class NeuralNetwork():
    #Basic structure of any NN
    numberOfInputs = int
    numberOfHidden = int
    numberOfOutputs = int
    #

    #Input to hidden Data holders
    inputs = []
    inputToHiddenWeights = [[]]
    inputToHiddenSum = []
    inputToHiddenOutput = []
    inputToHiddenBiases = []
    #

    #Hidden to Output Data Holders
    outputs = []
    hiddenToOutputWeights = [[]]
    hiddenToOutputSum = []
    hiddenToOutputBiases = []
    #

    #theta
    result = []
    #

    #Bp
    outputsErrorSignal = []
    hiddenErrorSignal  = []

    #Deltas
    inputToHiddenWeightsDelta = [[]]
    inputToHiddenBiasesDelta = []
    hiddenToOutputWeightsDelta = [[]]
    hiddenToOutputBiasesDelta = []
    #

    def __init__(self, numberOfInputs, numberOfHidden, numberOfOutputs):
        self.numberOfInputs = numberOfInputs
        self.numberOfHidden = numberOfHidden
        self.numberOfOutputs = numberOfOutputs

    def initialize(self):
        self.inputToHiddenWeights = [[0.0 for i in range(self.numberOfHidden)] for j in range(self.numberOfInputs)]
        self.inputs = [0.0 for i in range(self.numberOfInputs)]
        self.inputToHiddenSum = [0.0 for i in range(self.numberOfHidden)]
        self.inputToHiddenBiases = [0.0 for i in range(self.numberOfHidden)]
        self.inputToHiddenOutput = [0.0 for i in range(self.numberOfHidden)]

        self.hiddenToOutputWeights = [[0.0 for i in range(self.numberOfOutputs)] for j in range(self.numberOfHidden)]
        self.hiddenToOutputSum = [0.0 for i in range(self.numberOfOutputs)]
        self.hiddenToOutputBiases = [0.0 for i in range(self.numberOfOutputs)]
        self.outputs = [0.0 for i in range(self.numberOfOutputs)]
        self.result = [0.0 for i in range(self.numberOfOutputs)]

        self.inputToHiddenBiasesDelta = [0.0 for i in range(self.numberOfHidden)]
        self.inputToHiddenWeightsDelta = [[0.0 for i in range(self.numberOfHidden)] for j in range(self.numberOfInputs)]
        self.hiddenToOutputBiasesDelta = [0.0 for i in range(self.numberOfOutputs)]
        self.hiddenToOutputWeightsDelta = [[0.0 for i in range(self.numberOfOutputs)] for j in range(self.numberOfHidden)]

        self.outputsErrorSignal = [0.0 for i in range(self.numberOfOutputs)]
        self.hiddenErrorSignal = [0.0 for i in range(self.numberOfHidden)]

    def initializeWeights(self, weights):
        self.initialize()
        numberOfWeights = (self.numberOfInputs * self.numberOfHidden) + \
                          (self.numberOfHidden * self.numberOfOutputs) + \
                          self.numberOfHidden + \
                          self.numberOfOutputs
        it = iter(weights)

        if len(weights) != numberOfWeights:
            raise Exception("Number of weights isn't matched!")

        for i in range(0, self.numberOfInputs):
            for j in range(0, self.numberOfHidden):
                self.inputToHiddenWeights[i][j] = next(it)


            #print("Input to hidden weights -----> \n")
            #print(np.matrix(self.inputToHiddenWeights))

        for i in range(0, self.numberOfHidden):
            self.inputToHiddenBiases[i] = next(it)


        for i in range(0, self.numberOfHidden):
            for j in range(0, self.numberOfOutputs):
                self.hiddenToOutputWeights[i][j] = next(it)


        for i in range(0, self.numberOfOutputs):
            self.hiddenToOutputBiases[i] = next(it)


    def compute(self, inputArray):
        if len(inputArray) != self.numberOfInputs:
            raise Exception("Number of input isn't matched!")


        self.inputs = cp.copy(inputArray)

        self.inputToHiddenSum = [0.0 for i in range(self.numberOfHidden)]
        self.hiddenToOutputSum = [0.0 for i in range(self.numberOfOutputs)]

        #Compute input ti hidden weights sums
        for i in range(0, self.numberOfHidden):
            for j in range(0, self.numberOfInputs):
                self.inputToHiddenSum[i] += self.inputs[j] * self.inputToHiddenWeights[j][i]
        #

        #print("Input to hidden sum \n")
        #print(np.matrix(self.inputToHiddenSum))

        #Adding Biases
        for i in range(0, self.numberOfHidden):
            self.inputToHiddenSum[i] += self.inputToHiddenBiases[i]
        #

        #print("Input to hidden weights after adding biases -----> \n")
        #print(np.matrix(self.inputToHiddenSum))

        #calculate input to hidden output
        for i in range(0, self.numberOfHidden):
            self.inputToHiddenOutput[i] = Utilities().sigmoid(self.inputToHiddenSum[i])
        #

        #print("Input to hidden output sigmoid \n")
        #print(np.matrix(self.inputToHiddenOutput))

        #Compute hidden to output sum
        for i in range(0, self.numberOfOutputs):
            for j in range(0, self.numberOfHidden):
                self.hiddenToOutputSum[i] += self.inputToHiddenOutput[j] * self.hiddenToOutputWeights[j][i]
        #

        #adding biases
        for i in range(0, self.numberOfOutputs):
            self.hiddenToOutputSum[i] += self.hiddenToOutputBiases[i]
        #

        #print("Hidden to output weights after adding biases -----> \n")
        #print(np.matrix(self.hiddenToOutputSum))

        for i in range(0, self.numberOfOutputs):
            self.outputs[i] = Utilities().hyperTan(self.hiddenToOutputSum[i])

        print(self.outputs)

    def getError(self, trueTheta, output):
        sum = 0.0

        for i in range(0, self.numberOfOutputs):
            sum += (abs(trueTheta[i] - output[i]))

        return sum

    def backpropagation(self, trueTheta, learningRate, alpha, numOfIterations):
        if len(trueTheta) != self.numberOfOutputs:
            raise Exception("Number of Theta isn't matched!")

        currentError = self.getError(trueTheta, self.outputs)

        #Begin Algorithm
        for it in range(0, numOfIterations):
            if currentError <= 0.01:
                break

            print("====================\n Iteration number: ", it)

            #Output Layer
            for i in range(0, self.numberOfOutputs):
                tanhDerivative = (1 - self.outputs[i]) * (1 + self.outputs[i])
                self.outputsErrorSignal[i] = tanhDerivative * (trueTheta[i] - self.outputs[i])
            #
            #print(self.outputsErrorSignal)

            #Hidden Layer
            for i in range(0, self.numberOfHidden):
                sigmoidDerivative = (1 - self.inputToHiddenOutput[i]) * self.inputToHiddenOutput[i]
                sum = 0.0
                for j in range(0, self.numberOfOutputs):
                    sum += self.outputsErrorSignal[j] * self.hiddenToOutputWeights[i][j]
                self.hiddenErrorSignal[i] = sigmoidDerivative * sum
            #
            #print(self.hiddenErrorSignal)

            #Update Input to hidden weights
            for i in range(0, len(self.inputToHiddenWeights)):
                for j in range(0, len(self.inputToHiddenWeights[0])):
                    delta = (learningRate * self.hiddenErrorSignal[j] * self.inputs[i])
                    self.inputToHiddenWeights[i][j] += delta
                    self.inputToHiddenWeights[i][j] += alpha * self.inputToHiddenWeightsDelta[i][j]
                    self.inputToHiddenWeightsDelta[i][j] = delta
            #

            #print(self.inputToHiddenWeights)

            #Update Input to hidden Biases
            for i in range(0, self.numberOfHidden):
                delta = learningRate * self.hiddenErrorSignal[i] * 1.0
                self.inputToHiddenBiases[i] += delta
                self.inputToHiddenBiases[i] += alpha * self.inputToHiddenBiasesDelta[i]
                self.inputToHiddenBiasesDelta[i] = delta
            #

            #print(self.inputToHiddenBiases)

            #Update Hidden to output
            for i in range(0, len(self.hiddenToOutputWeights)):
                for j in range(0, len(self.hiddenToOutputWeights[0])):
                    delta = (learningRate * self.outputsErrorSignal[j] * self.inputToHiddenOutput[i])
                    self.hiddenToOutputWeights[i][j] += delta
                    self.hiddenToOutputWeights[i][j] += alpha * self.hiddenToOutputWeightsDelta[i][j]
                    self.hiddenToOutputWeightsDelta[i][j] = delta
            #

            #print(self.hiddenToOutputWeights)

            #Update hidden to output biases
            for i in range(0, self.numberOfOutputs):
                delta = learningRate * self.outputsErrorSignal[i] * 1.0
                self.hiddenToOutputBiases[i] += delta
                self.hiddenToOutputBiases[i] += alpha * self.hiddenToOutputBiasesDelta[i]
                self.hiddenToOutputBiasesDelta[i] = delta
            #
            #print(self.hiddenToOutputBiases)

            print("New Output :\n")
            #print(self.inputs)
            self.compute(self.inputs)
            currentError = self.getError(trueTheta, self.outputs)
            print("Error :\n", currentError)



    def run(self, weights, inputArray, trueTheta, learningRate, alpha, numOfIterations):
        self.initializeWeights(weights)
        self.compute(inputArray)
        self.backpropagation(trueTheta, learningRate, alpha, numOfIterations)
