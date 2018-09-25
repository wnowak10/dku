# ______________________________________________________________________________
# Imports.

import operator
import scipy 

import pandas as pd 
import numpy  as np

from initial_data_prep import df  # All training data.

# Statistical testing.
# ______________________________________________________________________________

# # Numeric variables:

num_vars = ['age',
			'wage',
			'capital_losses',
			'capital_gains',
			'num',
			'weeks',
			'instance',
			'dividends']
def build_rs(vars):
	"""
	Pearson correlations for numeric variables and `binary_income`.
	"""
	t_stats = {}  # A dictionary of t stats.
	for var in num_vars:
		t_stats[var] = scipy.stats.pearsonr(df[var], df['binary_income']) #, nan_policy = 'omit')[0]
	return t_stats

rs = build_rs(num_vars)
sorted_rs = sorted(rs.items(), key=operator.itemgetter(1))
print('Sorted pearson r stats:')
print('\n',sorted_rs, '\n')

# Categorical variables:
cat_vars = set(df.columns).difference(set(num_vars))
def build_chis(vars):
	"""
	Chi square test of two categorical variables.
	"""
	chi_coefs = {}  # A dictionary of Chi Square coefficients.
	for var in cat_vars:
		ct = pd.crosstab(df[var], df['binary_income'])
		ct_arr = np.asarray(ct)
		chi_coefs[var] = scipy.stats.chi2_contingency(ct_arr)[0]
	return chi_coefs

chi_coefs = build_chis(cat_vars)

sorted_chi = sorted(chi_coefs.items(), key=operator.itemgetter(1))
print('Sorted chi2 stats:')
print('\n',sorted_chi, '\n')