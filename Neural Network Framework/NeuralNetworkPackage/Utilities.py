__author__ = 'ahani'

import math


class Utilities():

    #Activation Functions
    def sigmoid(self, x):

        return (1.0 / (1.0 + math.exp(-x)))

    def hyperTan(self, x):
        return math.tanh(x);





