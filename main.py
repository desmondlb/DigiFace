import pandas as pd
import numpy as np
import re

lines = open('data/digitdata/trainingimages').read().splitlines()
lables = open('data/digitdata/traininglabels').read().splitlines()

test_lines = open('data/digitdata/testimages').read().splitlines()
test_lables = open('data/digitdata/testlabels').read().splitlines()

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

def calc_prior_distribution() -> dict:
    prior_prob = {}
    prior_prob_neg = {}

    for i in range(10):
        list_indices = np.where(lables==i)
        prior_prob[i] = len(digits[list_indices])/len(digits)
        prior_prob_neg[i] = len(np.delete(digits, [list_indices], 0))/len(digits)
        

    return prior_prob, prior_prob_neg
  
ldct_counts_y_true = {}
ldct_counts_y_false = {}

def calc_prob_x_given_y_raw(data_t, y_true, y_false, y):
    
    global ldct_counts_y_true, ldct_counts_y_false

    prob_f_given_y = {}
    prod_prob = 0
    prod_prob_y_false = 0
    for i in range(28):
        for j in range(28):
            count_zeroes = ldct_counts_y_true[y][(i,j)][0]
            count_ones = ldct_counts_y_true[y][(i,j)][1]
            # for data in y_true:
            #     if data[i][j] == 0:
            #         count_zeroes+=1
            #     else:
            #         count_ones+=1
            
            if data_t[i][j] == 1:
                if count_ones:
                    prod_prob += np.log10(count_ones/len(y_true))
                else:
                    prod_prob += 10**-4
            else:
                if count_zeroes:
                    prod_prob += np.log10(count_zeroes/len(y_true))
                else:
                    prod_prob += 10**-4

            
            count_zeroes = ldct_counts_y_false[y][(i,j)][0]
            count_ones = ldct_counts_y_false[y][(i,j)][1]
            # for data in y_false:
            #     if data[i][j] == 0:
            #         count_zeroes+=1
            #     else:
            #         count_ones+=1
            
            if data_t[i][j] == 1:
                if count_ones:
                    prod_prob_y_false += np.log10(count_ones/len(y_false))
                else:
                    prod_prob_y_false += 10**-4
        
            else:
                if count_zeroes:
                    prod_prob_y_false += np.log10(count_zeroes/len(y_false))
                else:
                    prod_prob_y_false += 10**-4


    
    # for i in range(28):
    #     for j in range(28):
            
                

    return prod_prob, prod_prob_y_false

def get_counts(y_true, y_false):
    features_y_true = {}
    features_y_false = {}
    for i in range(28):
        for j in range(28):
            count_zeroes = 0
            count_ones = 0
            for data in y_true:
                if data[i][j] == 0:
                    count_zeroes+=1
                else:
                    count_ones+=1

            features_y_true[(i,j)] = (count_zeroes, count_ones)

            count_zeroes = 0
            count_ones = 0
            for data in y_false:
                if data[i][j] == 0:
                    count_zeroes+=1
                else:
                    count_ones+=1

            features_y_false[(i,j)] = (count_zeroes, count_ones)

    return features_y_true, features_y_false


def calc_prob_test():
    
    global ldct_counts_y_true, ldct_counts_y_false

    for i in range(10):
        list_indices = np.where(lables==i)
        ldct_counts_y_true[i], ldct_counts_y_false[i] = get_counts(
            digits[list_indices], np.delete(digits, [list_indices], 0))


    final = []
    for index, data in enumerate(test_digits):
        possibilites = []
        for i in range(10):
            list_indices = np.where(lables==i)

            num_a, denom_a = calc_prob_x_given_y_raw(
                data, digits[list_indices], np.delete(digits, [list_indices], 0), i)

            val = (num_a + np.log10(prior_probabilities_y_true[i]))/(denom_a + np.log10(prior_probabilities_y_false[i]))

            possibilites.append(val)
        
        final.append(np.argmax(possibilites))

    return final

prior_probabilities_y_true,  prior_probabilities_y_false = calc_prior_distribution()
f = calc_prob_test()
print(len(digits))
    