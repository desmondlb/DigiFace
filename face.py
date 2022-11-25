# import pandas as pd
import numpy as np

lines = open('data/facedata/facedatatrain').read().splitlines()
lables = open('data/facedata/facedatatrainlabels').read().splitlines()

test_lines = open('data/facedata/facedatatest').read().splitlines()
test_lables = open('data/facedata/facedatatestlabels').read().splitlines()

faces = []
ptr1 = 0
while ptr1 != len(lines):
    face = lines[ptr1: ptr1+70]

    normalized_face = np.zeros(shape=(70,60), dtype=int)

    for index, line in enumerate(face):
        line = line.replace("+", "1").replace("#", "1").replace(" ", "0")
        normalized_face[index] = np.array(list(line), dtype=int)

    faces.append(normalized_face)
    
    ptr1 += 70

faces = np.array(faces)


test_faces = []
ptr1 = 0
while ptr1 != len(test_lines):
    digit = test_lines[ptr1: ptr1+70]

    normalized_face = np.zeros(shape=(70,60), dtype=int)

    for index, line in enumerate(digit):
        line = line.replace("+", "1").replace("#", "1").replace(" ", "0")
        normalized_face[index] = np.array(list(line), dtype=int)

    test_faces.append(normalized_face)
    
    ptr1 += 70

test_faces = np.array(test_faces)

lables = np.array(list(lables), dtype=int)
test_lables = np.array(list(test_lables), dtype=int)

def calc_prior_distribution() -> dict:
    prior_prob = {}
    prior_prob_neg = {}

    for i in range(2):
        list_indices = np.where(lables==i)
        prior_prob[i] = len(faces[list_indices])/len(faces)
        prior_prob_neg[i] = len(np.delete(faces, [list_indices], 0))/len(faces)
        

    return prior_prob, prior_prob_neg
  
ldct_counts_y_true = {}
ldct_counts_y_false = {}




def calc_prob_x_given_y_four_blocks(data_t, y_true, y_false, y, block_size):
    
    global ldct_counts_y_true, ldct_counts_y_false

    prob_f_given_y = {}
    prod_prob = 0
    prod_prob_y_false = 0
    for i in range(int(60/block_size)):
        for j in range(int(70/block_size)):


            dt_ones = np.count_nonzero(data_t[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
            if dt_ones:
                num_times_phi = 0
                for d in y_true:
                    x = np.count_nonzero(d[i*4:(i+1)*4, j*4:(j+1)*4] == 1)
                    if abs(x - dt_ones) <= 2:
                        num_times_phi += 1
                if num_times_phi:
                    prod_prob += np.log10(num_times_phi/len(y_true))
                else:
                    prod_prob += 10**-9

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
    for index, data in enumerate(test_faces):

        if index%100 == 0:
            print(index)
        possibilites = []
        for i in range(2):
            list_indices = np.where(lables==i)

            num_a, denom_a = calc_prob_x_given_y_four_blocks(
                data, faces[list_indices], np.delete(faces, [list_indices], 0), i, 5)

            # val = (num_a + np.log10(prior_probabilities_y_true[i]))/(denom_a + np.log10(prior_probabilities_y_false[i]))

            val = (num_a + np.log10(prior_probabilities_y_true[i]))

            possibilites.append(val)
        
        final.append(np.argmax(possibilites))

    return final

prior_probabilities_y_true,  prior_probabilities_y_false = calc_prior_distribution()
f = calc_prob_test()
f = np.array(f)

print(sum(1 for x,y in zip(test_lables,f) if x == y) / len(test_lables))
