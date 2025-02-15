# R-Fairness
This repository contains code for the experiments in "R-Fairness: Assessing Fairness of Ranking in Subjective Data".

To run our experiments first download the Yelp dataset from https://business.yelp.com/data/resources/open-dataset/ and copy the dataset "yelp_academic_dataset_business.json", "yelp_academic_dataset_review" and "yelp_academic_dataset_user.json" on the main directory.

Reproduce the plot for the Yelp dataset by running

python plot_yelp.py --attribute "years"
python plot_yelp.py --attribute "attitude"
python plot_yelp.py --attribute "reviews"
python plot_yelp.py --attribute "fans"

python plot_yelp_kendall.py --attribute "years"
python plot_yelp_kendall.py --attribute "attitude"
python plot_yelp_kendall.py --attribute "reviews"
python plot_yelp_kendall.py --attribute "fans"

The plot will be saved into the directory "data_boxplot"

The Amazon dataset will be provided upon request

The old plots for the old version of Yelp dataset can be found in the directory "Yelp_old_plot" 



