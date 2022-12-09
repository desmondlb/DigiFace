import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import math

class SVM():    
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
  