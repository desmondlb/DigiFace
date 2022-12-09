import numpy as np
from Perceptron import Perceptron
from statistics import stdev, mean
import matplotlib.pyplot as plt
import time

# Class to train Perceptron to classify faces
class FaceTrain:


    def __init__(self):
        self.height = 70
        self.width = 60



    #---------------------------------------------------#
    # 2. Load and extract features from file
    #---------------------------------------------------#
    def loadData(self, path):
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
            data.append(x)

        return data

    # ---------------------------------------------------#
    # Load labels from file
    # ---------------------------------------------------#
    def loadLabels(self, path):
        labels = [int(i) for i in open(path).read().splitlines()]
        return labels

    # ---------------------------------------------------#
    # 3. Function to randomly sample training data
    # ---------------------------------------------------#
    def rand_sampled_train_data(self, data, label, count):
        temp_data = []
        temp_label = []
        for i in range(count):
            j = np.random.randint(0, len(data))
            temp_data.append(data[j])
            temp_label.append(label[j])

        return np.array(temp_data), np.array(temp_label)


    # ---------------------------------------------------------------------------#
    # 3,4. Primary function which Creates & Trains Perceptron and Collects Stats
    # ---------------------------------------------------------------------------#
    def run(self):
        data_train_init = self.loadData(path="classification/facedata/facedatatrain")
        labels_train_init = self.loadLabels(path='classification/facedata/facedatatrainlabels')
        data_test = self.loadData(path='classification/facedata/facedatatest')
        labels_test = self.loadLabels(path='classification/facedata/facedatatestlabels')

        avg_accuracy = []
        stdev_accuracy = []
        runtimes = []
        partitions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Trains perceptron over different partitions of data
        # Collects statistics - average accuracy over 10 iterations,
        #                       standard deviation of accuracy over 10 iterations,
        #                       average training time over 10 iterations
        for percentData in partitions:

            # Limit on training data to be sampled
            limit = int(len(data_train_init) * percentData)

            acc_agg = []
            time_agg = []
            w = np.random.normal(0, 1, (2, 4201))

            #runs 10 iterations for percent partitions of data
            for j in range(10):
                start_time = time.time()
                print(">", end="")
                data_train, labels_train = self.rand_sampled_train_data(data_train_init, labels_train_init, limit)

                p = Perceptron(method='0', w=w)

                # agg_error and iter are used to calculate the weighted average of error
                agg_error = np.zeros(4201)
                iter = 0.0

                for i in range(limit):
                    iter += 0.1**i
                    delta = p.train(features=data_train[i], label=labels_train[i],
                                         prev_error=agg_error, iter=iter)

                    #aggregated error
                    for k in range(0, len(delta)):
                        agg_error[k] += delta[k]
                    agg_error *= 0.1

                # accuracy and runtime of each iteration are stored in following arrays -
                acc_agg.append(self.accuracy(p, data_test, labels_test, limit))
                time_agg.append(time.time() - start_time)

            print(percentData * 100, "% training data complete")
            avg_accuracy.append(mean(acc_agg))
            stdev_accuracy.append(stdev(acc_agg))
            runtimes.append(mean(time_agg))

        return avg_accuracy, stdev_accuracy, runtimes



    # ----------------------------------------------------------------#
    # 4. Function to calculate accuracy of classification on Test Data
    # ----------------------------------------------------------------#
    def accuracy(self, p, data, labels, limit):
        count = 0
        correct_count = 0

        for i in range(len(labels)):
            input = p.inputVector(features=data[i])
            guess = p.classify(input=input)
            if guess == labels[i]:
                correct_count += 1
            count += 1

        return (correct_count / count) * 100



if __name__ == '__main__':
    obj = FaceTrain()
    means, std_deviations, runtimes = obj.run()

    print("Avg accuracy : ", means)
    print("Runtimes : ", runtimes)
    print("Standard deviation: ", std_deviations)

    # Graph of Accuracy Stats
    plt.errorbar([i * 10 for i in range(1, 11)], means, yerr=std_deviations, ecolor='k', fmt='o', markersize=8,
                 capsize=6, color="r", linestyle="-")
    #plt.ylim(0, 100)
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Accuracy with Standard Deviation")
    plt.legend()
    plt.show()

    # Graph of Training Time
    plt.plot([i * 10 for i in range(1, 11)], runtimes)
    plt.xlabel("Percentage of training data (Faces)")
    plt.ylabel("Runtime of the Algorithm (Per Iteration)")
    plt.show()








