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

    def initializeWeights(self, weights):
        self.initialize()
        numberOfWeights = (self.numberOfInputs * self.numberOfHidden) + \
                          (self.numberOfHidden * self.numberOfOutputs) + \
                          self.numberOfHidden + \
                          self.numberOfOutputs
        index = 0
        it = iter(weights)

        if len(weights) != numberOfWeights:
            raise NameError("Number of weights isn't matched!")

        else:
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
            raise NameError("Number of input isn't matched!")

        else:
            self.inputs = cp.copy(inputArray)

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
                self.outputs[i] = math.tanh(self.hiddenToOutputSum[i])

            self.result = cp.copy(self.outputs)

            print(self.result)

    def getTheta(self):
        return self.result

    def run(self, weights, inputArray):
        self.initializeWeights(weights)
        self.compute(inputArray)
















