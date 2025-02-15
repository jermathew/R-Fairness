from argparse import ArgumentParser
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


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--attribute', type=str, choices=["sentiment", "gender"])
	args = parser.parse_args()
	attr = args.attribute


# Read the CSV file into a DataFrame
df_reviews = pd.read_csv("dataset/amazon_reviews.csv")


list_items = list(set(df_reviews["item_id"].tolist()))
num_items = len(list_items)



df_reviews['sentiment'] = df_reviews['sentiment'].astype(str)
df_reviews['rating'] = df_reviews['rating'].apply(lambda x: 5 if x > 5.1 else x)
df_reviews['rating'] = df_reviews['rating'].fillna(1).astype(int)
df_reviews['rating'] = df_reviews['rating'].astype(str)



# Add utility column with 0 as minimum

df_reviews['utility'] = None 
for i in range(len(df_reviews)):
    utility_i = eval(df_reviews.iloc[i]["counts"])["useful"]
    df_reviews.loc[i,"utility"] = utility_i + 1


def parse_into_dataframe(text_filepath: Path):

    with open(text_filepath, 'r') as file:
        data = file.read()

    data = data.split('\n')
    
    parsed_data = []
    for line in tqdm(data):
        if line:
            parsed_data.append(literal_eval(line))
    
    # create a dataframe
    df = pd.DataFrame(parsed_data, columns=column_names)
    return df



# Add gender to reviews
df_user = pd.read_csv("dataset/amazon_user_demographics.csv")
df_user[["user_id","gender"]]
dict_users = df_user.set_index('user_id').to_dict()['gender']
df_reviews['gender'] = df_reviews['user_id'].map(dict_users)
df_reviews['gender'] = df_reviews['gender'].astype(str)



# Add +1 in order to avoid 0 utility
df_reviews['utility'] = np.log2(df_reviews['utility'].astype(float)) +1



sentiment_dict = {
    'positive': 'pos',
    'negative': 'neg',
    'neutral': 'neu'
}

df_reviews['sentiment'] = df_reviews['sentiment'].map(sentiment_dict)


df_reviews = df_reviews.drop(columns=['date', 'text', 'country', 'counts'])

df_reviews.to_csv('dataset/amazon_review_final.csv', index=False)

# number_reviews is the number of reviews per object
number_reviews = 10000
number_kendall = 100


if attr == "sentiment":
    list_attr = list(set(df_reviews['sentiment']))
    list_attr = ["pos", "neg", "neu"]
    list_symbols = ["pos", "neg", "neu"]
    column_str = 'sentiment'

if attr == "gender":
    list_attr = list(set(df_reviews['gender']))
    list_attr = ["nan", "female", "male", "andy"]
    list_symbols = ["nan", "female", "male", "andy"]
    column_str = 'gender'

column_names = list_attr
num_attr = len(list_attr)
print(list_attr)



list_attr = [s for s in list_attr if s != "nan"]
list_symbols = [s for s in list_symbols if s != "nan"]
column_names = list_attr
num_attr = len(list_attr)



# For every value in list_attr, count the molteplicity for each item

attr_dict = {key: None for key in list_attr}

for attr_i in list_attr:
    multiplicity_i = 0
    for item in list_items:
        x = (df_reviews[column_str][df_reviews["item_id"] == item].head(50)).tolist()
        multiplicity = Counter(x)
        #print(multiplicity)
        multiplicity_i += multiplicity[attr_i]

    attr_dict[attr_i] = multiplicity_i


def divide_dict_values_by_sum(d):
    # Calculate the sum of all values in the dictionary
    total_sum = sum(d.values())

        
    # Divide each value in the dictionary by the total sum
    result_dict = {key: round((value / total_sum),4) for key, value in d.items()}
    
    return result_dict

# Divide all values of the dictionary by the sum of all values
frequencies = divide_dict_values_by_sum(attr_dict)

print(frequencies)

list_of_lists_items_exp_avg = list()
list_of_lists_items_exp_ratio = list()
#list_of_lists_items_treatment_avg = list()
list_of_lists_items_treatment_ratio = list()
list_of_lists_items_kendall = list()
for i in range(num_items):
        list_of_lists_items_exp_avg.append(list())
        list_of_lists_items_exp_ratio.append(list())
        #list_of_lists_items_treatment_avg.append(list())
        list_of_lists_items_treatment_ratio.append(list())
        list_of_lists_items_kendall.append(list())

for i in range(num_items):
        #print(i)
        item_i = list_items[i]
        df_reviews_i = df_reviews[df_reviews["item_id"] == item_i].head(number_reviews)
        df_reviews_i_kendall = df_reviews[df_reviews["item_id"] == item_i].head(number_kendall)

        list_df_attr_i = []
        list_df_attr_i_kendall = []
        for j in range(num_attr):
                list_df_attr_i.append(df_reviews_i[df_reviews_i[column_str] == list_attr[j]]) 
                list_df_attr_i_kendall.append(df_reviews_i_kendall[df_reviews_i_kendall[column_str] == list_attr[j]])



        for j in range(num_attr):
                if len(list_df_attr_i[j]) > 0:
                        list_of_lists_items_exp_avg[i].append(exposure_avg(list_df_attr_i[j]))
                        list_of_lists_items_exp_ratio[i].append(exposure_ratio(list_df_attr_i[j]))
                        #list_of_lists_items_treatment_avg[i].append(treatment_avg(list_df_attr_i[j]))
                        list_of_lists_items_treatment_ratio[i].append(treatment_ratio(list_df_attr_i[j]))
                else:
                        #print("NULL CASE", list_attr[j])
                        list_of_lists_items_exp_avg[i].append(0)
                        list_of_lists_items_exp_ratio[i].append(0)
                        #list_of_lists_items_treatment_avg[i].append(0)
                        list_of_lists_items_treatment_ratio[i].append(0)


                if len(list_df_attr_i_kendall[j]) > 0:
                        # KENDALL CASE: I NEED DF DIFFERENCE
                        merged_df = df_reviews_i_kendall.merge(list_df_attr_i_kendall[j], how='left', indicator=True)
                        difference_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')

                        if (len(difference_df) != 0):
                                list_of_lists_items_kendall[i].append(kendall_tau_asymmetric(list_df_attr_i_kendall[j], difference_df))
                else:
                        list_of_lists_items_kendall[i].append(-2)



with open('data_boxplot/one_level/list_of_lists_items_exp_avg_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_exp_avg:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')

with open('data_boxplot/one_level/list_of_lists_items_exp_ratio_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_exp_ratio:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')


with open('data_boxplot/one_level/list_of_lists_items_treatment_ratio_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_treatment_ratio:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')

with open('data_boxplot/one_level/list_of_lists_items_kendall_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_kendall:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')






# iterate through the text files and parse the data
fairness_measure_to_df = {}

for fairness_measure_name in ["exp_ratio", "treatment_ratio", "kendall", "exp_avg"]:
    filepath ='data_boxplot/one_level/list_of_lists_items_'+fairness_measure_name+"_"+str(attr)+".txt"
    fairness_measure_to_df[fairness_measure_name] = parse_into_dataframe(filepath)


# replace every -2 with NaN for the item_kendall fairness measure and
# every 0 with NaN for the other fairness measures
for fairness_measure, df in fairness_measure_to_df.items():
    if fairness_measure == 'kendall':
        fairness_measure_to_df[fairness_measure] = df.replace(-2, pd.NA)
    else:
        fairness_measure_to_df[fairness_measure] = df.replace(0, pd.NA)

# generate a boxplot for each fairness measure
# and save the plot as an html file
for fairness_measure, df in fairness_measure_to_df.items():

    print(fairness_measure)
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Box(y=df[column], name=column))
    
    # set the y-axis range to be between 0 and 1 except for the item_kendall fairness measure
    # where the range is between -1 and 1
    if fairness_measure == 'kendall':
        fig.update_layout(yaxis_range=[-1.1, 1.1])
        #fig.update_layout(title = fairness_measure)
    else:
        fig.update_layout(yaxis_range=[-0.1, 1.1])
        #fig.update_layout(title =  str(fairness_measure)+str(" for ") + str(column_str) + str(" with first ")+ str(number_reviews)+ " reviews")

    
    #fig.update_layout(showlegend=False,xaxis=dict(title='', showticklabels=False), width=300, height=250)
    fig.update_layout(showlegend=False, width=300, height=250)

    # Define the custom labels you want to use
    fig.update_xaxes(tickvals=list_attr, ticktext=list_symbols, tickfont=dict(size=22))
    fig.update_yaxes(tickfont=dict(size=22)) 

    fig.update_layout(margin=dict(l=5, r=5, t=5, b=0))

     
    
    fig.write_image("data_boxplot/plot/one_level/"+str(fairness_measure)+"_"+str(attr)+".png", scale=2)
    #fig.show()



