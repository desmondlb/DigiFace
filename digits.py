# import pandas as pd
import random
import statistics
import numpy as np
from statistics import stdev
from bayesian import Bayesian
from SVM import SVM
import math
import matplotlib.pyplot as plt
import time

class Faces(Bayesian,SVM):
    def __init__(self, feature_dims=None) -> None:

        self.feature_dims = feature_dims
        super().__init__()


if __name__ == '__main__':

    train_data_path = 'data/facedata/facedatatrain'
    train_label_path = 'data/facedata/facedatatrainlabels'
    test_data_path = 'data/facedata/facedatatest'
    test_label_path = 'data/facedata/facedatatestlabels'

    accuracies = {}
    std_deviation = {}
    runtimes = {}

    for i in range(1, 11):
        start_time = time.time()
        accuracy_random_split = []

        for j in range(10):
            obj = Faces(feature_dims={"HEIGHT":4,"WIDTH":4})
            obj.read_data(
                train_data_path=train_data_path, train_label_path=train_label_path, 
                test_data_path=test_data_path,test_label_path=test_label_path, train_percentage = i*0.1)

            obj.calc_prior_class_prob()

            obj.calc_feature_distribution(obj.feature_dims)

            f = obj.calc_prob_test(obj.feature_dims)
            f = np.array(f)


            accuracy_random_split.append(
                round((sum(1 for x,y in zip(obj.test_lables,f) if x == y) / len(obj.test_lables))*100,3))
        
        accuracies[i*10] = accuracy_random_split
        std_deviation[i*10] = round(stdev(accuracy_random_split),4)
        
        runtimes[i*10] = (time.time() - start_time)/10
        print(f"SD for {i*10}% Training Data is {round(stdev(accuracy_random_split),4)}")
        

    means = [sum(i)/len(i) for i in accuracies.values()]
    std_deviations = list(std_deviation.values())
    plt.errorbar([i*10 for i in range(1, 11)], means, yerr=std_deviations, ecolor='k', fmt='o', markersize=8, capsize=6, color="r", linestyle="-")
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Accuracy with Standard Deviation")
    plt.legend()
    plt.title("Naive Bayes")
    plt.show()

    plt.plot(list(runtimes.keys()),list(runtimes.values()))
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Runtime of the Algorithm (Per Iteration)")
    plt.title("Naive Bayes")
    plt.show()

    #SVM classifier
    runtimes.clear()
    start_time = time.time()
    accuracy = obj.classifier(obj.train_data,obj.train_lables,obj.test_data,obj.test_lables)
    iter = [10,20,30,40,50,60,70,80,90,100]
    obj_train_s = obj.train_data.shape[0]/100
    itr =[int(item * obj_train_s) for item in iter]
    acc = [[0]*10]*10
    std_arr = []
    std_acc = [] 
    mean_arr =[] 

    for i in range(0,10):   #for % of training data to use
        itr_data_train = []
        itr_data_train_labels=[]
        index = random.sample(range(0,obj.train_data.shape[0]),itr[i])
        for j in range(0,len(index)):
            itr_data_train.append(obj.train_data[index[j],:,:])
            itr_data_train_labels.append(obj.train_lables[index[j]])
        for k in range(0,10): # number of iterations to be done with % of training data
            acc[i][k]=obj.classifier(itr_data_train,itr_data_train_labels,obj.test_data,obj.test_lables)
        runtimes[i] = (time.time() - start_time)/10
        std_arr.append(statistics.stdev(acc[i]))
        mean_arr.append(statistics.fmean(acc[i])*100)
        print("On",iter[i],"percent data mean accuracy with SVM on faces is:",mean_arr[i],"\n")
        print("On",iter[i],"percent data standard deviation with SVM on faces is:",std_arr[i],"\n")

    plt.errorbar([i*10 for i in range(1, 11)], mean_arr, yerr=std_arr, ecolor='k', fmt='o', markersize=8, capsize=6, color="r", linestyle="-")
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Accuracy with Standard Deviation")
    plt.legend()
    plt.title("SVM")
    plt.show()

    plt.plot([i*10 for i in range(1, 11)],list(runtimes.values()))
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Runtime of the Algorithm (Per Iteration)")
    plt.title("SVM")
    plt.show()