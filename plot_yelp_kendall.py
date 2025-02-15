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
	parser.add_argument('--attribute', type=str, choices=["fans", "attitude", "years", "reviews"])
	args = parser.parse_args()
	attr = args.attribute


if attr == "fans":
    attr = "user_fans"
if attr == "attitude":
    attr = "user_attitude"
if attr == "years":
    attr = "user_years"
if attr == "reviews":
    attr = "user_reviews_count"


# Read the CSV file into a DataFrame
df_reviews = pd.read_csv("yelp_new_review.csv")

list_items = list(set(df_reviews["item_id"]))
num_items = len(list_items)

df_reviews.loc[df_reviews['utility'] == 0, 'utility'] += 1 


# utility function to parse data
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




# number_reviews is the number of reviews per restaurant
number_reviews = 10000
number_kendall = 100



if attr == "user_fans":
    list_attr = ['0', '1-5', '10+', '6-10']
    list_symbols = list_attr

if attr == "user_years":
    list_attr = ['6-10', '10+','0-5']
    list_symbols = list_attr

if attr == "user_reviews_count":
    list_attr = ["6-20", "1-5", "100-500", "21-50", "51-100", "500+"]
    list_symbols = list_attr

if attr == "user_attitude":
    list_attr = ["normal", "supporter", "hater"]
    list_symbols = list_attr


column_names = list_attr
column_str = attr
num_attr = len(list_attr)
print(list_attr)



# For every value in list_attr, count the molteplicity for each item

attr_dict = {key: None for key in list_attr}

for attr_i in list_attr:
    multiplicity_i = 0
    x = (df_reviews[column_str]).tolist()
    multiplicity = Counter(x)
    attr_dict[attr_i] = multiplicity[attr_i]




def divide_dict_values_by_sum(d):
    # Calculate the sum of all values in the dictionary
    total_sum = sum(d.values())
        
    # Divide each value in the dictionary by the total sum
    result_dict = {key: round((value / total_sum),4) for key, value in d.items()}
    
    return result_dict

# Divide all values of the dictionary by the sum of all values
frequencies = divide_dict_values_by_sum(attr_dict)

print(frequencies)


item_id_i = df_reviews.loc[0,"item_id"]
print(item_id_i)
first_i = 0
dict_first_last = dict()
for i in range(len(df_reviews)):
    if df_reviews.loc[i,"item_id"] != item_id_i:
        dict_first_last[item_id_i] = [first_i,i-1]
        item_id_i = df_reviews.loc[i,"item_id"]
        first_i = i

dict_first_last[item_id_i] = [first_i,i]


list_of_lists_items_kendall = list()
for i in range(num_items):
        list_of_lists_items_kendall.append(list())


for i in range(num_items):
        item_i = list_items[i]
        df_reviews_i = df_reviews.iloc[dict_first_last[item_i][0] : dict_first_last[item_i][1]].head(number_reviews)
        df_reviews_i_kendall = df_reviews_i.head(number_kendall)

        list_df_attr_i_kendall = []
        for j in range(num_attr):
                list_df_attr_i_kendall.append(df_reviews_i_kendall[df_reviews_i_kendall[column_str] == list_attr[j]])



        for j in range(num_attr):
                if len(list_df_attr_i_kendall[j]) > 0:
                        merged_df = df_reviews_i_kendall.merge(list_df_attr_i_kendall[j], how='left', indicator=True)
                        difference_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')

                        if (len(difference_df) != 0):
                                list_of_lists_items_kendall[i].append(kendall_tau_asymmetric(list_df_attr_i_kendall[j], difference_df))
                else:
                        list_of_lists_items_kendall[i].append(-2)


                

with open('data_boxplot/one_level/list_of_lists_items_kendall_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_kendall:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')




# iterate through the text files and parse the data
fairness_measure_to_df = {}
for fairness_measure_name in ["kendall"]:
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
    fig.update_xaxes(tickvals=list_attr, ticktext=list_symbols, tickfont=dict(size=18))
    fig.update_yaxes(tickfont=dict(size=18)) 

    fig.update_layout(margin=dict(l=5, r=5, t=5, b=0))
    
    fig.write_image("data_boxplot/plot/one_level/"+str(fairness_measure)+"_"+str(attr)+".png")
    #fig.show()


    









