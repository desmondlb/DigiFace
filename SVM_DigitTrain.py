import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
import time
import random
import statistics

class SVM_Digit():  

    def __init__(self) -> None:
        self.train_data = None
        self.train_lables = None
        self.test_data = None
        self.test_lables = None
        self.validation_data = None
        self.validation_lables = None

        self.data_height = None
        self.data_width = None

    def label_parser(self, raw_labels):

        return np.array(list(raw_labels), dtype=int)

    def data_parser(self, raw_data):
        data_points = []
        ptr = 0
        while ptr != len(raw_data):
            data_point = raw_data[ptr: ptr + self.data_height]

            normalized_data_points = np.zeros(shape=(self.data_height, self.data_width), dtype=int)

            for index, data_line in enumerate(data_point):
                data_line = data_line.replace("+", "1").replace("#", "1").replace(" ", "0")
                normalized_data_points[index] = np.array(list(data_line), dtype=int)

            data_points.append(normalized_data_points)
            
            ptr += self.data_height

        data_points = np.array(data_points)

        
        return data_points

    def read_data(
        self, train_data_path = None, train_label_path = None, 
        test_data_path = None, test_label_path = None, train_percentage = 1):
        
        train_data_raw = open(train_data_path).read().splitlines()
        train_labels_raw = open(train_label_path).read().splitlines()
        test_data_raw = open(test_data_path).read().splitlines()
        test_lables_raw = open(test_label_path).read().splitlines()

        self.data_height = len(train_data_raw)//len(train_labels_raw)
        self.data_width = len(train_data_raw[0])

        self.train_data = self.data_parser(train_data_raw)
        self.train_lables = self.label_parser(train_labels_raw)

        self.train_data, self.train_lables = shuffle(self.train_data, self.train_lables)
        self.train_data = self.train_data[:int(len(self.train_data)*train_percentage)]
        self.train_lables = self.train_lables[:int(len(self.train_lables)*train_percentage)]

        self.test_data = self.data_parser(test_data_raw)
        self.test_lables = self.label_parser(test_lables_raw)

    def classifier(self,train_data,train_labels,test_data,test_labels):
        classifier = SVC(kernel="linear")
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        if(train_data.ndim==3):
            x,y,z = train_data.shape   
        else:
            b,x,y,z= train_data.shape
        if train_labels.ndim==2:
            train_labels=np.squeeze(train_labels,axis=0)
        train_data = train_data.reshape(x,y*z)        
        x,y,z = test_data.shape
        test_data = test_data.reshape(x,y*z)          
        classifier.fit(train_data,train_labels)
        y = classifier.predict(test_data)
        accuracy=metrics.accuracy_score(test_labels,y)
        """confusion_matrix = metrics.confusion_matrix(test_labels, y)
        output = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
        output.plot()
        plt.show()
        """
        return accuracy
  
if __name__ == '__main__':
    train_data_path = 'data/digitdata/trainingimages'
    train_label_path = 'data/digitdata/traininglabels'
    test_data_path = 'data/digitdata/testimages'
    test_label_path = 'data/digitdata/testlabels'

    obj = SVM_Digit()
    obj.read_data(
                train_data_path=train_data_path, train_label_path=train_label_path, 
                test_data_path=test_data_path,test_label_path=test_label_path)
    runtimes1 = {}
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
        runtimes1[i] = (time.time() - start_time)/10
        std_arr.append(statistics.stdev(acc[i]))
        mean_arr.append(statistics.fmean(acc[i])*100)
        print("On",iter[i],"percent data mean accuracy with SVM on digits is:",mean_arr[i],"\n")
        print("On",iter[i],"percent data standard deviation with SVM on digits is:",std_arr[i],"\n")

    plt.errorbar([i*10 for i in range(1, 11)], mean_arr, yerr=std_arr, ecolor='k', fmt='o', markersize=8, capsize=6, color="g", linestyle="-",label="SVM")
    plt.xlabel("Percentage of training data (Digits)")
    plt.ylabel("Accuracy with Standard Deviation")
    plt.show()

    plt.plot([i*10 for i in range(1, 11)],list(runtimes1.values()))
    plt.xlabel("Percentage of training data (Digits)")
    plt.ylabel("Runtime of the Algorithm (Per Iteration)")
    plt.show()