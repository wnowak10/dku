import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# ______________________________________________________________________________
# Read column names.

def get_cols():
	with open('us_census_full/cols.txt', 'r') as f:
		x = f.readlines()

	cols = [a.split()[1] for a in x]
	cols.extend(('year','binary_income'))
	return cols

cols = get_cols()

# ______________________________________________________________________________
# Read data.

df = pd.read_csv('us_census_full/census_income_learn.csv', 
	header = None, 
	nrows = 50000,
	na_values = ' ?',
	names = cols)

test = pd.read_csv('us_census_full/census_income_test.csv', 
  header = None, 
  na_values = ' ?',
  names = cols)


def preprocess(df, edit = False):
	"""
	A function to preprocess US census data.

	edit - A boolean parameter controlling if we return non-raw data. For
	example, if edit set to `True`, we will take steps like setting
	seemingly unsensical data to NA. E.g. Someone with an hourly wage
	of $1000, who does not make over $50,000 in a year. This is possible,
	but rather unlikely, and likely confusing our algorithms, so we'll see 
	1000 to NaN instead.


	"""
	df['binary_income'] = np.where(df['binary_income'].str.contains('-'), 0, 1)
	y = df['binary_income']
	if edit:
		pass
	return df, y

df, y = preprocess(df)

test, test_y = preprocess(test)


