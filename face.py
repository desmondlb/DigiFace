# import pandas as pd
import numpy as np
from bayesian import Bayesian

class Faces(Bayesian):
    def __init__(self, feature_dims=None) -> None:

        self.feature_dims = feature_dims
        super().__init__()


if __name__ == '__main__':
    
    obj = Faces(feature_dims={"HEIGHT":4,"WIDTH":4})

    train_data_path = 'data/facedata/facedatatrain'
    train_label_path = 'data/facedata/facedatatrainlabels'
    test_data_path = 'data/facedata/facedatatest'
    test_label_path = 'data/facedata/facedatatestlabels'

    obj.read_data(
        train_data_path=train_data_path, train_label_path=train_label_path, 
        test_data_path=test_data_path,test_label_path=test_label_path)

    obj.calc_prior_class_prob()

    obj.calc_feature_distribution(obj.feature_dims)

    f = obj.calc_prob_test(obj.feature_dims)
    f = np.array(f)

    print(sum(1 for x,y in zip(obj.test_lables,f) if x == y) / len(obj.test_lables))
