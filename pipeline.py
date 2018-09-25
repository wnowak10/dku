# ______________________________________________________________________________
# Imports.

import pandas as pd 
import numpy  as np

from sklearn_pandas             import DataFrameMapper, CategoricalImputer
from sklearn.ensemble           import GradientBoostingClassifier
from sklearn.feature_selection  import SelectKBest
from sklearn.linear_model       import LogisticRegression, Ridge
from sklearn.metrics            import accuracy_score, median_absolute_error, make_scorer
from sklearn.model_selection    import GridSearchCV, KFold, RandomizedSearchCV, train_test_split 
from sklearn.pipeline           import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing      import MultiLabelBinarizer, Imputer, StandardScaler, LabelEncoder

from initial_data_prep          import df, y, cols, test, test_y


# ______________________________________________________________________________
# Data preparation.

(X_train, 
 X_test, 
 y_train, 
 y_test) = train_test_split(df, y, test_size=0.33, random_state=42)


# ______________________________________________________________________________
# MultiLabelBinarizer for categorical variables.

# def pre_MultiLabelBinarizer(vars, data):
#   """
#   Transform labels from strings to numbers. 
#   Output is an edit of X_train and X_test inplace to which 
#   we add columns. New columns are original names + '_encoded'.

#   This isn't really 'fitting' anything, so we aren't getting any 
#   leakage.

#   Annoying, inelegant solution. We need to fit transform all data with 
#   MultiLabelBinarizer so that labels remain constant in later test and 
#   train set transforms.
#   """
#   for var in vars:
#     m = DataFrameMapper([(var, [CategoricalImputer(), LabelEncoder()])],  df_out = True)
#     label_encoded = m.fit_transform(data)
#     # label_encoded = m.transform(data)
#     data[var+'_encoded'] = label_encoded
#     # X_test_label_encoded = m.transform(X_test)
#     # X_test[var+'_encoded'] = X_test_label_encoded
#   return data, m

# X_train = pre_MultiLabelBinarizer(['education'], X_train)
# X_test = pre_MultiLabelBinarizer(['education'], X_test)

# ______________________________________________________________________________
# Feature creation.

# ______________________________________________________________________________
# Imputation and feature selection.

mapper = DataFrameMapper([
                          ( ['education'],        [CategoricalImputer(), LabelEncoder()] ),
                          ( ['age'],              [Imputer(strategy='median'), StandardScaler()]),
                          ( ['wage'],              [Imputer(strategy='median'), StandardScaler()]),
                          ( ['weeks'],              [Imputer(strategy='median'), StandardScaler()]),
                          ( ['capital_gains'],     [Imputer(strategy='median'), StandardScaler()]),
                          ( ['capital_losses'],    [Imputer(strategy='median'), StandardScaler()]),
                          ( ['num'],              [Imputer(strategy='median'), StandardScaler()])
              			     ],  
                         df_out = True)

# After we have categorical encoded, fit_transform to prepare all train data.
fit_transformed_df = mapper.fit_transform(X_train)
X_np = np.asarray(fit_transformed_df)
# *Just* transform test.
X_np_test = np.asarray(mapper.transform(X_test))

y_np = np.asarray(y_train)  # y to np array for sklearn fit.

# ______________________________________________________________________________
# Sklearn pipeline instantiation

pipeline = Pipeline([
          ('SelectKBest', SelectKBest()),
          ('LogisticRegression', LogisticRegression())
          # ('Ridge', Ridge())
          # ('GradientBoostingClassifier', GradientBoostingClassifier(verbose = 1))
                    ])

# Possible values for GridSearchCV.
param_dist = {
							'SelectKBest__k': ['all']}
              # 'GradientBoostingClassifier__n_estimators': [10],#,50,100],
              # 'GradientBoostingClassifier__max_depth': [50],#,50,100]}
              # 'GradientBoostingClassifier__subsample': [.5]}

pipe = GridSearchCV(estimator = pipeline, 
                    param_grid=param_dist, 
                    cv=KFold(n_splits=4, random_state = 42),
                    n_jobs=1, # can set to -1 (num machine cores) if not using custom scorers.
                    verbose=1)

# ______________________________________________________________________________
# Sklearn pipeline instantiation

pipe.fit(X_np, y_np)

# Info about our fit:
print(pipe.best_params_)
# print(pipe.best_estimator_.named_steps['GradientBoostingClassifier'].feature_importances_)
train_preds = pipe.predict(X_np)
test_preds  = pipe.predict(X_np_test)

print('Test score is {0:.3f}'.format(accuracy_score(test_preds, y_test)))
print('Train score is {0:.3f}'.format(accuracy_score(train_preds, y_train)))


# print(pipe.best_estimator_)

# ______________________________________________________________________________
# Fit model on census_income_test.csv

mtt = mapper.transform(test)
test_np = np.asarray(mtt)

fully_held_out_test_preds = pipe.predict(test_np)

print('fully_held_out_test_preds score is {0:.3f}'.format(accuracy_score(fully_held_out_test_preds, test_y)))
