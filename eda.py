import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from initial_data_prep import df

# ______________________________________________________________________________
# Univariate analysis: Missingness

print('Percent of missing values per variable:')
print((df.isnull().sum(axis=0) / df.shape[0])*100, '\n')

# # ______________________________________________________________________________
# # Univariate analysis: Outliers

num_cols = ['age',
			'wage',
			'capital_losses',
			'capital_gains',
			'num',
			'weeks',
			'instance',
			'dividends']
for var in num_cols:
	normed = np.abs( df[var] - df[var].mean() ) / df[var].std()
	print('Outlier values for column {}.'.format(var))
	print(df[normed>3][var])
	# print(df[df['normed_'+var]<3].shape)

# ______________________________________________________________________________
# Univariate analysis: Plots


# # Numeric variables:
num_vars = ['age',
			'wage',
			'capital_losses',
			'capital_gains',
			'num',
			'weeks',
			'instance',
			'dividends',
			'binary_income']
for col in num_vars:
	# Distribution:
	sns.distplot(df[col])
	plt.show()
	# Covariance with target variable.
	sns.regplot(df[col], df['binary_income'])
	plt.show()

# Categorical variables:
cat_vars = list(set(df.columns).difference(set(num_vars)))
for i, col in enumerate(cat_vars):
	try:
		g = sns.countplot(x=col, 
						  hue = 'binary_income', 
					      data = df,
					      order = df[col].value_counts().index[1:])
		plt.xticks(rotation=90)
		plt.subplots_adjust(bottom=0.4)
		plt.savefig('{}.png'.format(i))
		# plt.show()
	except IndexError:
		print('Error with ', col)
		pass

