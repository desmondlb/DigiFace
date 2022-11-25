# import pandas as pd
import numpy as np

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

test_digits = np.array(test_digits[:200])

lables = np.array(list(lables), dtype=int)
test_lables = np.array(list(test_lables[:200]), dtype=int)

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


    return prod_prob, prod_prob_y_false


def get_feature(y, i, j, count_idx, feature_number):

    global ldct_counts_y_true, ldct_counts_y_false

    count_ones_list = []
    count_zero_list = []
    for m in range(i, i*4):
        for n in range(j, j*4):
            count_ones_list.append(ldct_counts_y_true[y][(i,j)][1])
            count_zero_list.append(ldct_counts_y_true[y][(i,j)][0])

    one = 0
    zero = 0
    if 1 in count_ones_list:
        one = 1
    if 0 in count_zero_list:
        zero = 1

    return zero, one


def calc_prob_x_given_y_four_blocks(data_t, y_true, y_false, y, block_size):
    
    global ldct_counts_y_true, ldct_counts_y_false

    prob_f_given_y = {}
    prod_prob = 0
    prod_prob_y_false = 0
    for i in range(int(28/block_size)):
        for j in range(int(28/block_size)):

            # count_zeroes = get_feature(y, i, j, 0, 4)
            # count_ones = get_feature(y, i, j, 0, 4)
            
            # count_zeroes = ldct_counts_y_true[y][(i,j)]
            # count_ones = ldct_counts_y_true[y][(i,j)]

            # if np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)/16 >= count_ones:
            # if count_ones!=0:
            dt_ones = np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            if dt_ones:
                num_times_phi = 0
                # x = ldct_counts_y_true[y][(i,j)]
                # if abs(x - dt_ones) <= 5:
                #     num_times_phi += 1
                for d in y_true:
                    x = np.count_nonzero(d[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
                    if abs(x - dt_ones) <= 5:
                        num_times_phi += 1
                if num_times_phi:
                    prod_prob += np.log10(num_times_phi/len(y_true))
                else:
                    prod_prob += 10**-9
            # else:
            #     prod_prob += 10**-9
            # else:
                # if count_zeroes:
                #     z = 4*4*len(y_true) - np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
                #     prod_prob += np.log10(z*len(y_true)/count_zeroes)
                # else:
                # prod_prob += 10**-4

            
                

            # if data_t[i][j] == 1:
            #     if count_ones:
            #         prod_prob += np.log10(count_ones/len(y_true))
            #     else:
            #         prod_prob += 10**-4
            # else:
            #     if count_zeroes:
            #         prod_prob += np.log10(count_zeroes/len(y_true))
            #     else:
            #         prod_prob += 10**-4

            
            # count_zeroes = ldct_counts_y_false[y][(i,j)][0]
            # count_ones = ldct_counts_y_false[y][(i,j)]

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
            # else:
                # if count_zeroes:
                #     z = 4*4*len(y_true) - np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
                #     prod_prob += np.log10(z*len(y_true)/count_zeroes)
                # else:
                # prod_prob += 10**-4

            # if data_t[i][j] == 1:
            #     if count_ones:
            #         prod_prob_y_false += np.log10(count_ones/len(y_false))
            #     else:
            #         prod_prob_y_false += 10**-4
        
            # else:
            #     if count_zeroes:
            #         prod_prob_y_false += np.log10(count_zeroes/len(y_false))
            #     else:
            #         prod_prob_y_false += 10**-4


    return prod_prob, prod_prob_y_false

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def check_if_ones(data, i, j, block_size):
    for block in split(data, block_size, block_size):
        pass


def get_counts(y_true, y_false, block_size = 1):
    features_y_true = {}
    features_y_false = {}
    for i in range(0, int(28/block_size)):
        for j in range(0, int(28/block_size)):
            count_zeroes = 0
            count_ones = 0
            for data in y_true:
                count_ones += np.count_nonzero(
                    data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] == 1
                    )
                count_zeroes += block_size*block_size - count_ones
                # if np.count_nonzero(
                #     data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] == 1
                #     ) < int(0.5*block_size**2):
                #     count_zeroes+=1
                # else:
                #     count_ones+=1

            denom = block_size*block_size*len(y_true)
            # features_y_true[(i,j)] = (count_zeroes/denom, count_ones/denom)
            features_y_true[(i,j)] = count_ones

            count_zeroes = 0
            count_ones = 0
            for data in y_false:
                count_ones += np.count_nonzero(
                    data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] == 1
                    )
                count_zeroes += block_size*block_size - count_ones
                # if np.count_nonzero(
                #     data[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] == 1
                #     ) < int(0.5*block_size**2):
                #     count_zeroes+=1
                # else:
                #     count_ones+=1

            denom = block_size*block_size*len(y_false)
            # features_y_false[(i,j)] = (count_zeroes/denom, count_ones/denom)
            features_y_false[(i,j)] = count_ones

    return features_y_true, features_y_false


def calc_prob_test():
    
    global ldct_counts_y_true, ldct_counts_y_false

    # for i in range(10):
    #     list_indices = np.where(lables==i)
    #     ldct_counts_y_true[i], ldct_counts_y_false[i] = get_counts(
    #         digits[list_indices], np.delete(digits, [list_indices], 0), block_size=4)


    final = []
    for index, data in enumerate(test_digits):

        if index%100 == 0:
            print(index)
        possibilites = []
        for i in range(10):
            list_indices = np.where(lables==i)

            # num_a, denom_a = calc_prob_x_given_y_raw(
            #     data, digits[list_indices], np.delete(digits, [list_indices], 0), i)

            num_a, denom_a = calc_prob_x_given_y_four_blocks(
                data, digits[list_indices], np.delete(digits, [list_indices], 0), i, 4)

            # val = (num_a + np.log10(prior_probabilities_y_true[i]))/(denom_a + np.log10(prior_probabilities_y_false[i]))

            val = (num_a + np.log10(prior_probabilities_y_true[i]))

            possibilites.append(val)
        
        final.append(np.argmax(possibilites))

    return final

prior_probabilities_y_true,  prior_probabilities_y_false = calc_prior_distribution()
f = calc_prob_test()
f = np.array(f)

print(sum(1 for x,y in zip(test_lables,f) if x == y) / len(test_lables))


    