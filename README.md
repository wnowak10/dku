# Write up


1. Setup:

```
$ conda create -n dku python=3.6

# source activate dku

$ pip install -r requirements.txt
```

2. To get final, assembled model accuracy, run:

`$ python ensemble.py`. 

which returns:

 >`Fully held out accuracy is 0.954.`
 
This file uses `pkl`ed predictions which are generated in `pipeline.py`.

2. To see "audit", see these files:

- `eda.py` for visualizations.
- `correlations.py` for bivariate statistical testing.

3. Some reflections on this challenge:


- This was a lot to take on in a quick period of time.  Apologies in advance for typos, model underperformance, lack of code prettiness, as the past and present week have been businer than anticipated....
- The data is imbalanced. As a result, I tried to built 2 models -- one on the entire training set, and one on a set where the `0` class was under sampled ... in theory allowing us to build a classifier which looked for more nuance in the `1` classes (as opposed to just predicting `0` each time, which would give ~93% accuracy). 
- I spend some time looking at plots and creating features, but more time and experimentation would likely improve things.
- I used GridSearchCV to optimize hyper parameters.
- I used a boosted decision tree as they tend to perform well with class imbalanced prediction problems, and they are fast to train, resistant to overfitting, and better able to handle non-linearities. I also tried logistic regression and random forests. I did not try to architect a neural network. 
- The `wage` column was very bothersome, as most entries were 0 and there were some very large values where the individual made less than #50K a year. I tried to address a bit of this confusion in `initial_data_prep.py`.
- Other features, too, are generated in `initial_data_prep.py`.

---------------

# Assignment

The following link lets you download an archive containing an “exercise” US Census dataset: http://thomasdata.s3.amazonaws.com/ds/us_census_full.zip

This US Census dataset contains detailed but anonymized information for approximately 300,000 people.

The archive contains 3 files: 
o A large learning .csv file
o Another test .csv file
o A metadata file describing the columns of the two above mentioned files (identical for both)

The goal of this exercise is to model the information contained in the last column (42nd), i.e., which people make more or less than $50,000 / year, from the information contained in the other columns. The exercise here consists of modeling a binary variable.

Work with R or Python to carry out the following steps:
o Import the learning and text files
o Based on the learning file:

o Make a quick statistic based and univariate audit of the different columns’ content and produce the results in visual / graphic format. This audit should describe the variable distribution, the % of missing values, the extreme values, and so on.
o Create a model using these variables (you can use whichever variables you want, or even create you own; for example, you could find the ratio or relationship between different variables, the one-hot encoding of “categorical” variables, etc.) to model wining more or less than $50,000 / year. Here, the idea would be for you to test one or two algorithms, such as logistic regression, or a decision tree. Feel free to choose others if wish.
o Choose the model that appears to have the highest performance based on a comparison between reality (the 42nd variable) and the model’s prediction. 
o Apply your model to the test file and measure it’s real performance on it (same method as above).

The goal of this exercise is not to create the best or the purest model, but rather to describe the steps you took to accomplish it.

Explain areas that may have been the most challenging for you.

Find clear insights on the profiles of the people that make more than $50,000 / year. For example, which variables seem to be the most correlated with this phenomenon?

Finally, you push your code on GitHub to share it with me, or send it via email.

Once again, the goal of this exercise is not to solve this problem, but rather to spend a few hours on it and to thoroughly explain your approach.