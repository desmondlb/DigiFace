from abc import *
import numpy as np

class Bayesian(ABC):
    def __init__(self) -> None:
        self.train_data = None
        self.train_lables = None
        self.test_data = None
        self.test_lables = None
        self.validation_data = None
        self.validation_lables = None

        self.data_height = None
        self.data_width = None

        self.class_wise_feature_distribution_table = list()

        self.prior_probabilities_y_true = None
        self.prior_probabilities_y_false = None
    

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


    def calc_prior_class_prob(self):
        prior_prob = {}
        prior_prob_neg = {}

        for label in np.unique(self.train_lables):
            list_indices = np.where(self.train_lables==label)
            prior_prob[label] = len(self.train_data[list_indices])/len(self.train_data)
            prior_prob_neg[label] = len(np.delete(self.train_data, [list_indices], 0))/len(self.train_data)
            

        self.prior_probabilities_y_true = prior_prob
        self.prior_probabilities_y_false = prior_prob_neg


    def calc_feature_distribution(self, feature_dims):

        for label in np.unique(self.train_lables):
            table = np.zeros(shape=(
                (self.data_height//feature_dims['HEIGHT'])*(self.data_width//feature_dims['WIDTH']),
                1+feature_dims['HEIGHT']*feature_dims['WIDTH']), 
                dtype=int)

            list_indices = np.where(self.train_lables==label)

            feature_counter = 0
            for i in range(self.data_height//feature_dims['HEIGHT']):
                for j in range(self.data_width//feature_dims['WIDTH']):
                    for digit in self.train_data[list_indices]:
                        value = np.count_nonzero(
                            digit[i*feature_dims['HEIGHT']:(i+1)*feature_dims['HEIGHT'], 
                            j*feature_dims['WIDTH']:(j+1)*feature_dims['WIDTH']] == 1)

                        table[feature_counter][value] += 1

                    feature_counter += 1

            self.class_wise_feature_distribution_table.append(table)



    def calc_prob_x_given_y_four_blocks(self, data_t, y_true, y, feature_dims):

        prod_prob = 0
        feature_counter = 0
        for i in range(int(self.data_height/feature_dims['HEIGHT'])):
            for j in range(int(self.data_width/feature_dims['WIDTH'])):

                dt_ones = np.count_nonzero(
                    data_t[i*feature_dims['HEIGHT']:(i+1)*feature_dims['HEIGHT'], 
                    j*feature_dims['WIDTH']:(j+1)*feature_dims['WIDTH']] == 1)

                if dt_ones:

                    num_times_phi = self.class_wise_feature_distribution_table[y][feature_counter][dt_ones]

                    if num_times_phi:
                        prod_prob += np.log10(num_times_phi) + np.log10(num_times_phi/len(y_true))

                feature_counter += 1

        return prod_prob


    def calc_prob_test(self, feature_dims):
    
        final = []
        for index, data in enumerate(self.test_data):

            possibilites = []
            for label in np.unique(self.train_lables):
                list_indices = np.where(self.train_lables==label)

                num_a = self.calc_prob_x_given_y_four_blocks(
                    data, self.train_data[list_indices], label, feature_dims)

                # val = (num_a + np.log10(prior_probabilities_y_true[i]))/(denom_a + np.log10(prior_probabilities_y_false[i]))

                val = (num_a + np.log10(self.prior_probabilities_y_true[label]))

                possibilites.append(val)
            
            final.append(np.argmax(possibilites))

        return final

    def read_data(
        self, train_data_path = None, train_label_path = None, 
        test_data_path = None, test_label_path = None):
        
        train_data_raw = open(train_data_path).read().splitlines()
        train_labels_raw = open(train_label_path).read().splitlines()
        test_data_raw = open(test_data_path).read().splitlines()
        test_lables_raw = open(test_label_path).read().splitlines()

        self.data_height = len(train_data_raw)//len(train_labels_raw)
        self.data_width = len(train_data_raw[0])

        self.train_data = self.data_parser(train_data_raw)
        self.train_lables = self.label_parser(train_labels_raw)

        self.test_data = self.data_parser(test_data_raw)
        self.test_lables = self.label_parser(test_lables_raw)