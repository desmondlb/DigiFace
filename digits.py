import numpy as np
from bayesian import Bayesian

class Digits(Bayesian):
    def __init__(self, feature_dims=None) -> None:

        self.feature_dims = feature_dims
        super().__init__()

if __name__ == '__main__':
    
    

    train_data_path = 'data/digitdata/trainingimages'
    train_label_path = 'data/digitdata/traininglabels'
    test_data_path = 'data/digitdata/testimages'
    test_label_path = 'data/digitdata/testlabels'

    accuracies = []

    for i in range(1, 11):
        obj = Digits(feature_dims={"HEIGHT":4,"WIDTH":1})
        obj.read_data(
            train_data_path=train_data_path, train_label_path=train_label_path, 
            test_data_path=test_data_path,test_label_path=test_label_path, train_percentage = i*0.1)

        obj.calc_prior_class_prob()

        obj.calc_feature_distribution(obj.feature_dims)

        f = obj.calc_prob_test(obj.feature_dims)
        f = np.array(f)


        accuracies.append(sum(1 for x,y in zip(obj.test_lables,f) if x == y) / len(obj.test_lables))
    
    print(accuracies)
