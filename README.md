# R-Fairness: Assessing Fairness of Ranking in Subjective Data
This repository contains the code and data for the paper *"R-Fairness: Assessing Fairness of Ranking in Subjective Data"*, which was accepted at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).

### Abstract
Subjective data, reflecting individual opinions, permeate collaborative rating platforms like Yelp and Amazon, influencing everyday decisions. Despite the prevalence of such platforms, little attention has been given to fairness in their context, where groups of reviewers writing best-ranked reviews for best-ranked items have more influence on users' behavior. In this paper, we design and evaluate a new framework for the assessment of fairness of rankings for different reviewer groups in collaborative rating platforms. The key contributions are evaluating group exposure for different queries and platforms and comparing how various fairness definitions behave in different settings. Experiments on real datasets reveal insights into the impact of item ranking on fairness computation and the varying robustness of these measures.

### Getting started
To run our experiments, first download the Yelp dataset from https://business.yelp.com/data/resources/open-dataset/. Then copy the `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review` and `yelp_academic_dataset_user.json` files of the dataset to the main directory of this repository. To preprocess the dataset to reproduce the results in the paper, run the following command:

```bash
python prepare_dataset.py
```

To reproduce all the experimental results and plots in the paper, run the following commands:

```bash

python plot_yelp.py --attribute "years" 

python plot_yelp.py --attribute "attitude"

python plot_yelp.py --attribute "reviews"

python plot_yelp.py --attribute "fans"

python plot_yelp_kendall.py --attribute "years"

python plot_yelp_kendall.py --attribute "attitude"

python plot_yelp_kendall.py --attribute "reviews"

python plot_yelp_kendall.py --attribute "fans"

```

The `--attribute` argument specifies the attribute to be used for the analysis. The available attributes are `years`, `attitude`, `reviews`, and `fans`. The script will generate plots for each attribute and save them in the directory `data_boxplot`. 

### Note
For the Amazon dataset, please contact the authors directly. The dataset is not publicly available due to size limitations. The authors will provide the dataset upon request.



