import numpy as np
from scipy.special import softmax
import math


#-----------------------------------------------#
# 1. Class implements Perceptron
#-----------------------------------------------#

class Perceptron:

    # perceptron algorithm has a weight vector for each class
    def __init__(self, method, w):
        self.weights = w
        self.method = method



    # creates an input vector from features
    def inputVector(self, features):
        input = []
        input.append(1)
        for i in features:
            input.append(i)

        return np.array(input)


    # classifies input vector
    def classify(self, input):
        score = np.empty(len(self.weights))
        for i in range(len(self.weights)):
            score[i] = np.matmul(self.weights[i], input)


        # ACTIVATION FUNCTION-- normalize the data
        
        minn = min(score)
        maxx = max(score)
        for i in range(len(score)):
            if(maxx == minn):
                score[i] = 0.5
            else:
                score[i] = (score[i] - minn)/(maxx - minn)

        # ALTERNATE ACTIVATION FUNCTION - Softmax
        # return np.random.choice(a=list(range(0, len(score))), p=softmax(score))
        # return np.argmax(softmax(score))

        return np.argmax(score)



    # returns error for each node, total error, and expected classification
    def calculate_error(self, guess, target):
        expected = self.classToVector(len(self.weights[0]),target)
        guess_vector = self.classToVector(len(self.weights[0]),guess)
        error = target - guess
        return expected - guess_vector, error, target



    # Trains Perceptron from a choice of 3 update strategies
    def train(self, features, label, prev_error, iter):
        input = self.inputVector(features)
        score = self.classify(input)
        delta, error, expected = self.calculate_error(score, label)
        if self.method == '0':
            updt_rate = np.full(len(self.weights[0]), 1)    # update rate = 1 for all variables
        elif self.method == 'delta':
            updt_rate = delta
        elif self.method == 'weighted avg delta':
            updt_rate = (delta+prev_error)/iter
        if error != 0:
            for j in range(len(self.weights)):
                for i in range(len(self.weights[0])):
                    if j == expected:
                        self.weights[j][i] += abs(updt_rate[j]) * input[i]
                    else:
                        self.weights[j][i] -= abs(updt_rate[j]) * input[i]

        return delta


    # converts label to a vector (used to calculate node-wise error)
    def classToVector(self, num_of_classes, target):
        result_vector = np.full(num_of_classes, 0)
        result_vector[target] = 1
        return result_vector

if __name__ == '__main__':
    arr = list(range(10))
    print(arr)








