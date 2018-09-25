import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from initial_data_prep import df

# ______________________________________________________________________________
# Univariate analysis: Missingness

print(df.isnull().sum(axis=0) / df.shape[0])

# ______________________________________________________________________________
# Univariate analysis: Outliers
# (only really a thing in numeric variables).

num_cols = ['wage']
for var in num_cols:
	df['normed_'+var] = np.abs( df[var] - df[var].mean() ) / df[var].std() 
	print(df[df['normed_'+var]<3].shape)

# ______________________________________________________________________________
# Univariate analysis: Plots

# Categorical variables:
g = sns.countplot(df['class'])
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.6)
plt.show()


# Numeric variables:
numeric_vars = ['wage', 'weeks', 'capital_gains']
for col in numeric_vars:
	# sns.distplot(df[col])
	# plt.show()
	# sns.boxplot(df['binary_income'], df[col])
	# plt.show()
	sns.regplot(df[col], df['binary_income'])
	plt.show()


