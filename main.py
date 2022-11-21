import pandas as pd
import numpy as np
import re

lines = open('data/digitdata/trainingimages').read().splitlines()
lables = open('data/digitdata/traininglabels').read().splitlines()

digits = []
ptr1 = 0
ptr2 = 0
while ptr2 != len(lines):
    digit = lines[ptr2: ptr2+28]

    normalized_digit = []

    for line in digit:
        line = line.replace("+", "1").replace("#", "1").replace(" ", "0")
        normalized_digit.append(line)

    digits.append(normalized_digit)
    
    ptr2 += 28

df = pd.DataFrame(
    {'Data': digits,
     'Label': lables,
    })

def calc_prior_distribution() -> dict:
    prior_prob = {}

    for i in range(10):
        # ans = df.groupby('Label')['Label'].transform('count')
        # ans = df['Label'].value_counts()
        prior_prob[df.Label.value_counts().index[i]] = df['Label'].value_counts().iloc[i]/len(df)
        

    return prior_prob


prior_probabilities_y = calc_prior_distribution()
def get_image_features(image):

    #Basic
    features = {}
    for i in range(28):
        for j in range(28):
            features[(i, j)] = int(image[i][j])
    return features


def prob_feature_given_y(pos, feature, y):
    

    # ans = df.groupby('Label')['Label'].transform('count')
    # ans = df['Label'].value_counts()

    df_data_with_given_label = df.loc[df['Label']==str(y)]
    count = 0
    for index, row in df_data_with_given_label.iterrows():
        features = get_image_features(row.Data)

        if features[(pos)] == feature:
            count += 1

    return count/len(df_data_with_given_label)
    

def calc_prob_x_given_y(features):
    
    prob_f_given_y = {}
    for i in range(10):
        prod_prob = 1
        for k, feature in features.items():
            prod_prob *= prob_feature_given_y(k, feature, i)
        
        prob_f_given_y[i] = prod_prob

    return prod_prob

def calc_prob_features():
    prob_x_given_y = {}

    for i in range(10):
        # ans = df.groupby('Label')['Label'].transform('count')
        df_data_with_given_label = df.loc[df['Label']==str(i)]

        for index, row in df_data_with_given_label.iterrows():
            features = get_image_features(row.Data)

            prob_x_given_y[i] = calc_prob_x_given_y(features)

calc_prob_features()
print(len(digits))
    