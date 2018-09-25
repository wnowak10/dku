# ______________________________________________________________________________
# Imports.

import scipy 

import pandas as pd 
import numpy  as np

from initial_data_prep import df  # All training data.

# Statistical testing.
# ______________________________________________________________________________

print(df.info())
# # Numeric variables:

num_vars = ['age',
			'wage',
			'capital_losses',
			'capital_gains',
			'num',
			'weeks']
def build_ts(vars):
	t_stats = {}  # A dictionary of t stats.
	for var in num_vars:
		t_stats[var] = scipy.stats.ttest_ind(df[var], df['binary_income'], nan_policy = 'omit')[0]
	return t_stats

t_stats = build_ts(num_vars)

print(t_stats)
print(scipy.stats.ttest_ind(df['capital_gains'], df['binary_income'], nan_policy = 'omit'))
print(scipy.stats.ttest_ind(df['weeks'], df['binary_income'], nan_policy = 'omit'))

# Categorical variables:
cat_vars = set(df.columns).difference(set(num_vars))
def build_chis(vars):
	chi_coefs = {}  # A dictionary of Chi Square coefficients.
	for var in cat_vars:
		ct = pd.crosstab(df[var], df['binary_income'])
		ct_arr = np.asarray(ct)
		chi_coefs[var] = scipy.stats.chi2_contingency(ct_arr)[0]
	return chi_coefs

chi_coefs = build_chis(cat_vars)
print('\n',chi_coefs)