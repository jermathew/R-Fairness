import pandas as pd
from fairness_all_functions import * 
import random
from ast import literal_eval
from pathlib import Path
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import json
import matplotlib.pyplot as plt
from collections import Counter



with open('yelp_academic_dataset_review.json') as f:
    data = '[' + ','.join(f.readlines()) + ']'

df_review = pd.read_json(data)

with open('yelp_academic_dataset_user.json') as f:
    data = '[' + ','.join(f.readlines()) + ']'

df_users = pd.read_json(data)

# REVIEWERS

df_users = df_users[['user_id', 'review_count', 'yelping_since', "useful", "fans", "average_stars"]]
dict_users = dict(zip(df_users['user_id'], df_users.index))
df_users = df_users[df_users["review_count"] > 0]



# Define bins and labels
bins = [-1, 5, 20, 50, 100, 500, float('inf')]  # -1 to include 0 in the first bin
labels = ['1-5', '6-20', '21-50', '51-100', '100-500', "500+"]

# Create the new column with ranges
df_users['REVIEW_COUNT'] = pd.cut(df_users['review_count'], bins=bins, labels=labels)


# Define bins and labels
bins = [-1, 0, 5, 10, float('inf')]  # -1 to include 0 in the first bin
labels = ['0', '1-5', '6-10', '10+']

# Create the new column with ranges
df_users['FANS'] = pd.cut(df_users['fans'], bins=bins, labels=labels)


# Define bins and labels
bins = [-1, 2, 4, float('inf')]  # -1 to include 0 in the first bin
labels = ['hater', 'normal', 'supporter']

# Create the new column with ranges
df_users['ATTITUDE'] = pd.cut(df_users['average_stars'], bins=bins, labels=labels)


df_users['AGE'] = df_users['yelping_since'].str[:4]
df_users['AGE'] = 2022 - (df_users['AGE']).astype(int)


# Define bins and labels
bins = [-1, 5, 10, float('inf')]  # -1 to include 0 in the first bin
labels = ['0-5', '6-10', '10+']

# Create the new column with ranges
df_users['YEARS'] = pd.cut(df_users['AGE'], bins=bins, labels=labels)
df_users = df_users.drop("AGE", axis = 1)


df_users.to_csv('yelp_new_users.csv', index=False)



with open('yelp_academic_dataset_review.json') as f:
    data = '[' + ','.join(f.readlines()) + ']'

df_review = pd.read_json(data)


# ITEMS/REVIEWS


df_review = df_review.drop(["funny", "cool", "text"], axis = 1)
df_review = df_review.rename(columns={'useful': 'utility'})
df_review["utility"] = (df_review["utility"]).astype(int) +1
df_review.loc[df_review['utility'] == 0, 'utility'] += 1
df_review = df_review.rename(columns={'business_id': 'item_id'})
df_review = df_review.sort_values(by = "item_id")



df_reviews["user_reviews_count"] = None
df_reviews["user_fans"] = None
df_reviews["user_attitude"] = None
df_reviews["user_years"] = None

dict_users = dict(zip(df_users['user_id'], df_users.index))





for index, row in df_reviews.iterrows():
    user_i = row[1]
    if index % 100000 == 0:
        print(round(index/len(df_reviews),2))
    if user_i in dict_users.keys():
        user_index = dict_users[user_i]
        df_user_i = df_users.iloc[user_index]
        review_i_count = df_user_i["REVIEW_COUNT"]
        #print(review_i_count)
        review_i_fans = df_user_i["FANS"]
        review_i_attitude = df_user_i["ATTITUDE"]
        review_i_years = df_user_i["YEARS"]
        df_reviews.loc[index, "user_reviews_count"] = review_i_count
        df_reviews.loc[index, "user_fans"] = review_i_fans
        df_reviews.loc[index, "user_attitude"] = review_i_attitude
        df_reviews.loc[index, "user_years"] = review_i_years
    else:
        df_reviews.loc[index, "user_reviews_count"] = None
        df_reviews.loc[index, "user_fans"] = None
        df_reviews.loc[index, "user_attitude"] = None
        df_reviews.loc[index, "user_years"] = None


df_reviews_final = df_reviews[df_reviews["user_reviews_count"].notna()]


df_reviews_final["position"] = None



item_current = "0000000000000000000"
position = 0
for index, row in df_reviews_final.iterrows():
    if index % 100000 == 0:
        print(round(index/len(df_reviews_final),2))
    if row["item_id"] == item_current:
        df_reviews_final.loc[index, "position"] = position
        position += 1
    else:
        df_reviews_final.loc[index, "position"] = 0
        position = 1
        item_current = row["item_id"]


df_reviews_final["position"] = df_reviews_final["position"] +1


df_reviews_final.to_csv('yelp_new_review.csv', index=False)





