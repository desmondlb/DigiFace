import numpy
import numpy as np


class Perceptron:

    def __init__(self, num_of_classes, size):
        #self.weights = np.random.uniform(0, 1, (4201,1))
        #self.weights = np.random.normal(0, 1, (4201, 1))
        #self.weights_1 =np.random.normal(0,1,size)
        #self.weights_0 =np.random.normal(0,1,size)
        self.weights = np.random.normal(0,1,(num_of_classes,size))
        #self.weights = np.zeros((num_of_classes,size))
        #self.weights = np.full((num_of_classes,size),-1)



        '''
        self.input = []
        if features:
            self.input.append([1])
            for i in features:
               self.input.append([i])
            self.input = np.array(self.input)
        '''


    def inputVector(self, features):
        input = []
        if features:
            input.append(1)
            for i in features:
                input.append(i)
            input = np.array(input)

        '''
        if features:
            input.append([1])
            for i in features:
                input.append([i])
            input = np.array(input)
        '''
        return input


    def classify(self, input):
        #score = np.matmul(self.weights.T, self.input
        score = numpy.empty(len(self.weights))
        #print(score)
        for i in range(len(self.weights)):
            #print(np.matmul(self.weights[i], input))
            score[i] = np.matmul(self.weights[i], input)
        #print(score)
        #score_0 = np.matmul(self.weights[0], input)

        #activation function
        return numpy.argmax(score)


        #return score

    def calculate_error(self, guess, target):
        return target-guess

    def train(self, features, label, prev_error, iter):
        input = self.inputVector(features)
        score = self.classify(input)
        error = self.calculate_error(score, label)
        avg_err = (error+prev_error)/iter
        #if error != 0:
        for i in range(4201):


            '''
            if error > 0:
                self.weights[1][i] += input[i]
                self.weights[0][i] -= input[i]
            if error < 0:
                self.weights[1][i] -= input[i]
                self.weights[0][i] += input[i]
            '''


            if error > 0:
                self.weights[1][i] += abs(avg_err) * input[i]
                self.weights[0][i] -= abs(avg_err) * input[i]
            if error < 0:
                self.weights[1][i] -= abs(avg_err) * input[i]
                self.weights[0][i] += abs(avg_err) * input[i]



        return error





if __name__ == '__main__':
    #features = [1, 0, 0, 1]
    obj = Perceptron(num_of_classes=2, size=5)
    input = [0,1,0,0,1]
    print(obj.weights)
    #print(len(obj.weights))
    print(obj.classify(input))

    #print(obj.weights.T)
    #print(obj.x)
    #print(np.matmul(obj.weights.T, obj.x))


