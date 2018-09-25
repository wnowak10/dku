# ______________________________________________________________________________
# Imports.

import pickle

import pandas as pd 
import numpy  as np

from sklearn.metrics            import accuracy_score, median_absolute_error, make_scorer

from initial_data_prep          import df, y, cols, test, test_y, TRAINING_SET_MEAN

# ______________________________________________________________________________
# Read saved predictions.

with open('balanced_preds.pkl', 'rb') as f:
	balanced_preds = pickle.load(f)

with open('test_preds.pkl', 'rb') as f:
	test_preds = pickle.load(f)

# ______________________________________________________________________________
# Constants

W1 = 1  # Alas, these are best performing parameters :/.
W2 = 0
W3 = 0

# Array of percent of training data that are 0, 1s.
mean_arr = [1-TRAINING_SET_MEAN, TRAINING_SET_MEAN]

# ______________________________________________________________________________
# Combine predictions in linear combination. 3 contributions are original predictions,
# predictions on a balanced data set, and raw class means.

weighted_full_preds       = np.asarray([i * (W1) for i in test_preds])
weighted_subsampled_preds = np.asarray([i * (W2) for i in balanced_preds])
weighted_means            = np.asarray([i * (W3) for i in mean_arr])

combination = weighted_full_preds + weighted_subsampled_preds + weighted_means

print('Fully held out accuracy is {0:.3f}'.format(accuracy_score(combination.argmax(axis =1), test_y)))