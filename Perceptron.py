import numpy as np

class Perceptron:

    def __init__(self, method, w):
        self.weights = w
        self.method = method



    def inputVector(self, features):
        input = []
        #if features:
        input.append(1)
        for i in features:
            input.append(i)

        return np.array(input)


    def classify(self, input):
        score = np.empty(len(self.weights))
        for i in range(len(self.weights)):
            score[i] = np.matmul(self.weights[i], input)
        minn = min(score)
        maxx = max(score)
        #normalize the data
        for i in range(len(score)):
            if(maxx == minn):
                score[i] = 0.5
            else:
                score[i] = (score[i] - minn)/(maxx - minn)

        #activation function
        return np.argmax(score)


    def calculate_error(self, guess, target):
        expected = self.classToVector(len(self.weights[0]),target)
        guess_vector = self.classToVector(len(self.weights[0]),guess)
        error = target - guess
        return expected - guess_vector, error, target



    def train(self, features, label, prev_error, iter):
        input = self.inputVector(features)
        score = self.classify(input)
        #error, expected = self.calculate_error(score, label)
        delta, error, expected = self.calculate_error(score, label)
        #avg_err = (error+prev_error)/iter
        if self.method == '0':
            updt_rate = np.full(len(self.weights[0]),1)
        elif self.method == 'delta':
            updt_rate = delta
        elif self.method == 'weighted avg delta':
            updt_rate = (delta+prev_error)/iter
        #avg_err = (delta+prev_error)/iter
        #print(len(self.weights[0]))
        if error != 0:
            for j in range(len(self.weights)):
                for i in range(len(self.weights[0])):
                    if j == expected:
                        self.weights[j][i] += abs(updt_rate[j]) * input[i]
                    else:
                        self.weights[j][i] -= abs(updt_rate[j]) * input[i]

        return delta

    def classToVector(self, num_of_classes, target):
        result_vector = np.full(num_of_classes,0)
        result_vector[target] = 1
        return result_vector





if __name__ == '__main__':
    # return 0





