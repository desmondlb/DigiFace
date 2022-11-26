# import pandas as pd
import numpy as np

lines = open('data/digitdata/trainingimages').read().splitlines()
lables = open('data/digitdata/traininglabels').read().splitlines()

test_lines = open('data/digitdata/testimages').read().splitlines()
test_lables = open('data/digitdata/testlabels').read().splitlines()

class_wise_feature_distribution_table = None

digits = []
ptr1 = 0
while ptr1 != len(lines):
    digit = lines[ptr1: ptr1+28]

    normalized_digit = np.zeros(shape=(28,28), dtype=int)

    for index, line in enumerate(digit):
        line = line.replace("+", "1").replace("#", "1").replace(" ", "0")
        normalized_digit[index] = np.array(list(line), dtype=int)

    digits.append(normalized_digit)
    
    ptr1 += 28

digits = np.array(digits)


test_digits = []
ptr1 = 0
while ptr1 != len(test_lines):
    digit = test_lines[ptr1: ptr1+28]

    normalized_digit = np.zeros(shape=(28,28), dtype=int)

    for index, line in enumerate(digit):
        line = line.replace("+", "1").replace("#", "1").replace(" ", "0")
        normalized_digit[index] = np.array(list(line), dtype=int)

    test_digits.append(normalized_digit)
    
    ptr1 += 28

test_digits = np.array(test_digits)

lables = np.array(list(lables), dtype=int)
test_lables = np.array(list(test_lables), dtype=int)

def calc_prior_class_prob() -> dict:
    prior_prob = {}
    prior_prob_neg = {}

    for i in range(10):
        list_indices = np.where(lables==i)
        prior_prob[i] = len(digits[list_indices])/len(digits)
        prior_prob_neg[i] = len(np.delete(digits, [list_indices], 0))/len(digits)
        

    return prior_prob, prior_prob_neg
  
ldct_counts_y_true = {}
ldct_counts_y_false = {}

def calc_prob_x_given_y_four_blocks(data_t, y_true, y, block_size):
    
    global class_wise_feature_distribution_table

    prod_prob = 0
    prod_prob_y_false = 0
    feature_counter = 0
    for i in range(int(28/block_size[0])):
        for j in range(int(28/block_size[1])):

            dt_ones = np.count_nonzero(
                data_t[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] == 1)

            if dt_ones:

                num_times_phi = class_wise_feature_distribution_table[y][feature_counter][dt_ones]
                # num_times_phi = 0
                # for d in y_true:
                #     x = np.count_nonzero(
                #         d[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] == 1)
                #     if abs(x - dt_ones) <= 4:
                #         num_times_phi += 1
                if num_times_phi:
                    prod_prob += np.log10(num_times_phi) + np.log10(num_times_phi/len(y_true))

            feature_counter += 1
                # else:
                #     prod_prob += 0
                    
            # dt_ones = np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            # if dt_ones:
            #     num_times_phi = 0
            #     for d in y_true:
            #         x = np.count_nonzero(d[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            #         if abs(x - dt_ones) <= 4:
            #             num_times_phi += 1
            #     if num_times_phi:
            #         prod_prob += np.log10(num_times_phi/len(y_true))
            #     else:
            #         prod_prob += 10**-9

            # if count_ones!=0:
            # dt_ones = np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            # if dt_ones:
            #     num_times_phi = 0
            #     for d in y_false:
            #         x = np.count_nonzero(d[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            #         if x == dt_ones:
            #             num_times_phi += 1
            #     if num_times_phi:
            #         prod_prob_y_false += np.log10(num_times_phi/len(y_false))
            #     else:
            #         prod_prob_y_false += 10**-9
            # else:
            #     prod_prob_y_false += 10**-9

            # if np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)/16 >= count_ones:
            #     if count_ones:
            #         prod_prob += np.log10(np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)*len(y_false)/count_ones)
            #     else:
            #         prod_prob += 10**-4


    return prod_prob, prod_prob_y_false


def calc_prob_test():
    
    global ldct_counts_y_true, ldct_counts_y_false

    # for i in range(10):
    #     list_indices = np.where(lables==i)
    #     ldct_counts_y_true[i], ldct_counts_y_false[i] = get_counts(
    #         digits[list_indices], np.delete(digits, [list_indices], 0), block_size=4)


    final = []
    for index, data in enumerate(test_digits):

        possibilites = []
        for i in range(10):
            list_indices = np.where(lables==i)

            num_a, denom_a = calc_prob_x_given_y_four_blocks(
                data, digits[list_indices], i, (4, 1))

            # val = (num_a + np.log10(prior_probabilities_y_true[i]))/(denom_a + np.log10(prior_probabilities_y_false[i]))

            val = (num_a + np.log10(prior_probabilities_y_true[i]))

            possibilites.append(val)
        
        final.append(np.argmax(possibilites))

    return final


def calc_feature_distribution(block_size) -> list:

    class_wise_feature_distribution = []

    for i in range(10):
        table = np.zeros(shape=((28//block_size[0])*(28//block_size[1]),1+block_size[0]*block_size[1]), dtype=int)
        list_indices = np.where(lables==i)

        feature_counter = 0
        for j in range(28//block_size[0]):
            for k in range(28//block_size[1]):
                for digit in digits[list_indices]:
                    value = np.count_nonzero(
                        digit[j*block_size[0]:(j+1)*block_size[0], k*block_size[1]:(k+1)*block_size[1]] == 1)

                    table[feature_counter][value] += 1

                feature_counter += 1

        class_wise_feature_distribution.append(table)
        

    return class_wise_feature_distribution

prior_probabilities_y_true,  prior_probabilities_y_false = calc_prior_class_prob()

class_wise_feature_distribution_table = calc_feature_distribution((4,1))

f = calc_prob_test()
f = np.array(f)

print(sum(1 for x,y in zip(test_lables,f) if x == y) / len(test_lables))
