import numpy as np
from Perceptron import Perceptron
from statistics import stdev
import json
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


class FaceTrain:
    #write the code to import data as pixels

    def __init__(self):
        self.height = 70
        self.width = 60
        self.count = 0
        #data = [[0]*width for i in range (height)]
        #self.data_train = []
        #self.data_label


    #read of file returns an empty string at EOF


    def loadData(self, path):
        #################### LOAD TRAINING DATA###########################
        temp = open(path).readlines()

        data = []
        k = 0
        while True:
            x = []
            if k >= len(temp):
                break
            for i in range(k+0, k+self.height):
                for j in range(0, self.width):
                    if temp[i][j] == "#":
                        x.append(1)
                    else:
                        x.append(0)
                k += 1
            #print("processing- " ,k)
            data.append(x)


        #print(len(data_train))
        return data




    '''
    for i in range(0,height):
        for j in range(0,width):
            print(x[i][j], end="")
        print()
    '''

    '''
    print(len(data_train[449]))
    
    k=0
    for i in range(0,height):
        for j in range (0, width):
            print(data_train[2][k], end="")
            k+=1
        print()
    '''


    def loadLabels(self, path):
        ###############################LOAD TRAINING LABELS##############################
        labels = []
        labels = [int(i) for i in open(path).read().splitlines()]
        #print(len(labels))
        #print(labels)
        return labels

    def writeInitialWeights(self, w):

        with open('starting_points.txt', 'w') as f:
            x = json.loads(w)
            f.write(json.dumps(x))


    def run(self, percentData):
        data_train = self.loadData(path="classification/facedata/facedatatrain")
        labels_train = self.loadLabels(path='classification/facedata/facedatatrainlabels')
        data_test = self.loadData(path='classification/facedata/facedatatest')
        labels_test = self.loadLabels(path='classification/facedata/facedatatestlabels')
        limit = int(451*percentData)
        acc_agg = []
        print(len(data_train))
        #p = Perceptron()

        #print(p.weights_1)
        #print(p.lr)
        #print(len(labels_train))

        '''
        print("Before Training")
        print("Guess | Actual")
        for i in range(401,421):
            input = p.inputVector(features= data_train[i])
            guess = p.classify(input=input)
            print(guess," | ",labels_train[i])
        '''

        # compare other samples after training-
        '''
        print("After Training")
        print("Guess | Actual")
        for i in range(401, 421):
            input = p.inputVector(features=data_train[i])
            guess = p.classify(input=input)
            print(guess, " | ", labels_train[i])
        '''

        for j in range(1):
            print(">", end="")
            p = Perceptron(2, 4201)
            print(p.weights)
            #self.writeInitialWeights(p.weights)
            agg_error = 0.0
            iter = 0.0
            #acc_agg.append(obj.run(0.1))
            #print(data_train[:45])
            for i in range(limit):
                iter += 0.1**i
                agg_error += p.train(features=data_train[i], label=labels_train[i],
                                     prev_error=agg_error, iter=iter)
                agg_error *= 0.1
            print(p.weights)
            acc_agg.append(self.accuracy(p,data_test,labels_test, limit))
            print(acc_agg)




        return acc_agg
        #return self.accuracy(p,data_train,labels_train, limit)

    def accuracy(self, p, data, labels, limit):

        count = 0
        correct_count = 0
        #for i in range(limit,451):
        for i in range(len(labels)):
            input = p.inputVector(features=data[i])
            guess = p.classify(input=input)
            if guess == labels[i]:
                correct_count += 1
            count += 1


        #print("Accuracy avg: ", acc_agg/10)
        return (correct_count/count)*100






if __name__ == '__main__':
    obj = FaceTrain()

    acc_agg = obj.run(1)
    print("Avg accuracy : ", sum(acc_agg)/1)
    print("Standard deviation: ", stdev(acc_agg))
    print("Accuracy Range : ", max(acc_agg) - min(acc_agg))










