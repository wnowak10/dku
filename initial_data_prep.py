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
	TRAINING_SET_MEAN = y.mean()
	if edit:
		# If they make over $100 / hr and less than $50k a year, assume this $100/hr is a mistake.
		df['wage'] = np.where(((df['wage'] > 100) & (df['capital_gains'] == 0) & (df['capital_gains'] == 0)), df['wage'].median(), df['wage'])
		# df['wage'] = np.where(((df['wage'] > 50) & (df['binary_income'] == 0)), df['wage'].median(), df['wage'])
		# df['wage'] = np.where( ( (df['wage']==0) & (df['binary_income'] == 1) ), df['wage'].mean(), df['wage'])
		df['ttl_clcltd_ernings'] = 40*df['wage']*df['weeks']

		df['professional'] = np.where(df['majoroc'].str.contains('Professional|Executive|Sales|Precision',regex = True), 1,0)
		df['white'] = np.where(df['mace'].str.contains('White|Asian',regex = True), 1,0)
		df['ad_degree'] = np.where(df['education'].str.contains('Bachelors|Doctorate|Masters|Prof',regex = True), 1,0)
		df['younger22'] = np.where(df['age']<22, 1,0)
		df['older65'] = np.where(df['age']>65, 1,0)
		df['occupation2'] = np.where(df['occupation'] == 2, 1,0)
		
		# df['log_cgs'] = np.log(df['capital_gains'])

		df['jointu65_tax'] = np.where(df['tax'] == 'Joint both under 65', 1,0)
		# df['employed?'] = np.where(df['full'] == 'Full-time schedules', 1, 0)
	return df, y, TRAINING_SET_MEAN

df, y, TRAINING_SET_MEAN = preprocess(df, edit = True)

# print(df[df['wage2']>100][['wage', 'binary_income']])
# print(df['wage2'].isnull())
test, test_y, _ = preprocess(test, edit = True)


