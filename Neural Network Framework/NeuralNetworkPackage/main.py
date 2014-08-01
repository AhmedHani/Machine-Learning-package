import numpy as np
from NeuralNetwork import *
import random

NUM_OF_INPUTS = 3
NUM_OF_HIDDEN = 4
NUM_OF_OUTPUTS = 2
THETA = [-0.8500, 0.7500]
LEARNING_RATE = 0.89
ALPHA = 0.1
NUM_OF_ITERATIONS = 1000

def main():
    weights = [round(random.uniform(-1, 1), 2) for i in range(0, NUM_OF_INPUTS * NUM_OF_HIDDEN + NUM_OF_HIDDEN * NUM_OF_OUTPUTS + NUM_OF_HIDDEN + NUM_OF_OUTPUTS)]
    print("Weights:\n", np.matrix(weights), "\n")

    inputs = [round(random.random(), 2) for i in range(0, NUM_OF_INPUTS)]
    print("Inputs:\n", np.matrix(inputs), "\n")

    print("Output:\n")
    NeuralNetwork(NUM_OF_INPUTS, NUM_OF_HIDDEN, NUM_OF_OUTPUTS).run(weights, inputs, THETA, LEARNING_RATE, ALPHA, NUM_OF_ITERATIONS)


main()