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
#print(num_items)

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


# number_reviews is the number of reviews per item
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

attr_dict


# ONE LEVEL

item_id_i = df_reviews.loc[0,"item_id"]
first_i = 0
dict_first_last = dict()
for i in range(len(df_reviews)):
    if df_reviews.loc[i,"item_id"] != item_id_i:
        dict_first_last[item_id_i] = [first_i,i-1]
        item_id_i = df_reviews.loc[i,"item_id"]
        first_i = i

dict_first_last[item_id_i] = [first_i,i]



list_of_lists_items_exp_avg = list()
list_of_lists_items_exp_ratio = list()
list_of_lists_items_treatment_ratio = list()
list_of_lists_items_kendall = list()
for i in range(num_items):
        list_of_lists_items_exp_avg.append(list())
        list_of_lists_items_exp_ratio.append(list())
        list_of_lists_items_treatment_ratio.append(list())
        list_of_lists_items_kendall.append(list())


for i in range(num_items):
        #print(i)
        item_i = list_items[i]
        df_reviews_i = df_reviews.iloc[dict_first_last[item_i][0] : dict_first_last[item_i][1]].head(number_reviews)

        list_df_attr_i = []
        for j in range(num_attr):
                list_df_attr_i.append(df_reviews_i[df_reviews_i[column_str] == list_attr[j]]) 



        for j in range(num_attr):
                if len(list_df_attr_i[j]) > 0:
                        list_of_lists_items_exp_avg[i].append(exposure_avg(list_df_attr_i[j]))
                        list_of_lists_items_exp_ratio[i].append(exposure_ratio(list_df_attr_i[j]))
                        list_of_lists_items_treatment_ratio[i].append(treatment_ratio(list_df_attr_i[j]))
                else:
                        list_of_lists_items_exp_avg[i].append(0)
                        list_of_lists_items_exp_ratio[i].append(0)
                        list_of_lists_items_treatment_ratio[i].append(0)
                



with open('data_boxplot/one_level/list_of_lists_items_exp_avg_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_exp_avg:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')

with open('data_boxplot/one_level/list_of_lists_items_exp_ratio_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_exp_ratio:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')

with open('data_boxplot/one_level/list_of_lists_items_treatment_ratio_'+str(attr)+'.txt', 'w') as file:
    for sublist in list_of_lists_items_treatment_ratio:
        file.write('[' + ', '.join(map(str, sublist)) + ']\n')



# iterate through the text files and parse the data
fairness_measure_to_df = {}

#for fairness_measure_name in ["exp_ratio", "treatment_ratio", "kendall"]:
for fairness_measure_name in ["exp_avg", "exp_ratio", "treatment_ratio"]:
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

     
    
    fig.write_image("data_boxplot/plot/one_level/"+str(fairness_measure)+"_"+str(attr)+".png", scale=2)
    #fig.show()


# TWO LEVELS

    

with open('yelp_academic_dataset_business.json') as f:
    data = '[' + ','.join(f.readlines()) + ']'

df_item = pd.read_json(data)




city = "Santa Barbara"
target = "italian"

if (True):

    df_selected = df_item[(df_item["city"] == city)]
    df_selected = df_selected[df_selected["categories"].str.contains(target, case=False, na=False)]
    #df_selected = df_selected[df_selected["attributes"].str.contains("RestaurantsPriceRange2", case=False, na=False)]
    #print(len(df_selected))
    df_selected = df_selected.reset_index()
    df_selected = df_selected.drop("index", axis = 1)
    df_selected.head(3)

    index_to_remove = list()
    for index,row in df_selected.iterrows():
        if row["attributes"]:
            if "RestaurantsPriceRange2" in row["attributes"].keys():
                x = 1 
            else:
                index_to_remove.append(index)
        else:
            index_to_remove.append(index)

    df_selected = df_selected.drop(index_to_remove)
    df_selected = df_selected.reset_index()

    df_selected["price"] = None
    for index,row in df_selected.iterrows():
        price_i = row["attributes"]["RestaurantsPriceRange2"]
        df_selected.loc[index, "price"] = price_i

    df_selected = df_selected.drop("index", axis = 1)

    df_selected["distance"] = None
    for index,row in df_selected.iterrows():
        lat_i = row["latitude"]
        lon_i = row["longitude"]
        dist_i = math.sqrt((lat_i - 34.420830)**2 + (lon_i+119.698189)**2)
        df_selected.loc[index, "distance"] = dist_i

    #df_selected = df_selected.drop("level_0", axis = 1)

    df_query = pd.DataFrame(columns=["position", "item_id", "ordering_attribute"])



    for ordering_attribute in ["price_asc", "price_disc", "distance", "stars", "review_count"]:
        if ordering_attribute in ["price_asc", "price_disc", "distance"]:
            if ordering_attribute == "price_asc":
                df_selected = df_selected.sort_values(by="price", ascending = True)
            if ordering_attribute == "price_disc":
                df_selected = df_selected.sort_values(by="price", ascending = False)
            if ordering_attribute == "star":
                df_selected = df_selected.sort_values(by=ordering_attribute, ascending = True)
        else:
            df_selected = df_selected.sort_values(by=ordering_attribute, ascending = False) # decrescente

        for index, item_id in enumerate(df_selected["business_id"]):
            df_query.loc[len(df_query)] = [index, item_id, ordering_attribute]


k = len(df_query[df_query["ordering_attribute"] == "stars"])
p = len(set(df_query["ordering_attribute"]))
#print(k,p)

list_attributes = ["price_asc", "price_disc", "distance", "stars", "review_count"]
number_reviews_two_levels = 15



# Dataframe 2 levels
if True:
    exp_ratio_2_levels = list() 
    treatment_ratio_2_levels = list() 
    for j in list_attributes:
            exp_ratio_2_levels.append(list())
            treatment_ratio_2_levels.append(list())


    for i, attr in enumerate(list_attributes):

        permutation_items = df_query[df_query["ordering_attribute"] == attr]["item_id"].tolist()

        list_of_lists_items_exp_ratio = list()
        list_of_lists_items_treatment_ratio = list()
        for j in range(k):
                list_of_lists_items_exp_ratio.append(list())
                list_of_lists_items_treatment_ratio.append(list())


        for z in range(len(permutation_items)):
            item = permutation_items[z]
            df_i = df_reviews[df_reviews["item_id"] == item].head(number_reviews_two_levels)

            list_df_attr_i = []
            for j in range(num_attr):
                    list_df_attr_i.append(df_i[df_i[column_str] == list_attr[j]]) 


            for j in range(num_attr):
                if len(list_df_attr_i[j]) > 0:
                        list_of_lists_items_exp_ratio[z].append(exposure_ratio(list_df_attr_i[j]))
                        list_of_lists_items_treatment_ratio[z].append(treatment_ratio(list_df_attr_i[j]))
                else:
                        list_of_lists_items_exp_ratio[z].append(0)
                        list_of_lists_items_treatment_ratio[z].append(0)

        normalization_factor = 0
        for w in range(k):
                normalization_factor += (1/(math.log(w+2,2)))

        for x in range(num_attr):
            count = 0
            for w in range(k):
                count += (1/(math.log(w+2,2))) * list_of_lists_items_exp_ratio[w][x]
            exp_ratio_2_levels[i].append(count/normalization_factor)

        for x in range(num_attr):
            count = 0
            for w in range(k):
                count += (1/(math.log(w+2,2))) * list_of_lists_items_treatment_ratio[w][x]
            treatment_ratio_2_levels[i].append(count/normalization_factor)  

# Save the dataframes
if True:
    with open("data_boxplot/two_levels/list_of_lists_items_exp_ratio_"+str(number_reviews_two_levels)+".txt", 'w') as file:
        for sublist in exp_ratio_2_levels:
            file.write('[' + ', '.join(map(str, sublist)) + ']\n')

    with open("data_boxplot/two_levels/list_of_lists_items_treatment_ratio_"+str(number_reviews_two_levels)+".txt", 'w') as file:
        for sublist in treatment_ratio_2_levels:
            file.write('[' + ', '.join(map(str, sublist)) + ']\n')


fairness_measure_to_df = {}
# Create the fairness_measures
if True:
    for fairness_measure_name in ["exp_ratio", "treatment_ratio"]:
        filepath = "data_boxplot/two_levels/list_of_lists_items_"+fairness_measure_name+"_"+str(number_reviews_two_levels)+".txt"
        fairness_measure_to_df[fairness_measure_name] = parse_into_dataframe(filepath)

    # replace 0 with NaN for the  fairness measures
    for fairness_measure, df in fairness_measure_to_df.items():
        fairness_measure_to_df[fairness_measure] = df.replace(0, pd.NA)       



#fig, ax = plt.subplots(2, 1)
#plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
for measure in ["exp_ratio", "treatment_ratio"]:
    df = fairness_measure_to_df[measure]

    sublist = (list(df_query["ordering_attribute"]))[::k]
    df['color'] = sublist

    column_names_map = dict(zip(list_attr, list_symbols))
    # Rename columns
    df = df.rename(columns=column_names_map)
    df = df.dropna(axis=1, how='all')

    plt.figure(figsize=(10, 4))
    plt.tight_layout()
    #plt.update_layout(margin=dict(l=5, r=5, t=5, b=0))

    plt.xticks(fontsize=18)  # Aumenta la dimensione dei tick dell'asse X
    plt.yticks(fontsize=18)  # Aumenta la dimensione dei tick dell'asse Y


    pd.plotting.parallel_coordinates(df, 'color', color=("#000000","#009292","#ff6db6","#b6dbff", "#920000","#db6d00","#24ff24","#ffff6d"))
    plt.savefig('data_boxplot/two_levels/'+measure+"_"+str(number_reviews_two_levels)+'.pdf', bbox_inches='tight')
    #plt.show()











