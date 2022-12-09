import numpy as np
from statistics import stdev
from bayesian import Bayesian
import matplotlib.pyplot as plt
import time

class Faces(Bayesian):
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
    plt.show()

    plt.plot(list(runtimes.keys()),list(runtimes.values()))
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Runtime of the Algorithm (Per Iteration)")
    plt.show()