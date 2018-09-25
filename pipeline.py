# ______________________________________________________________________________
# Imports.

import argparse
import pickle
import sys
import warnings

import pandas as pd 
import numpy  as np

from sklearn_pandas             import DataFrameMapper, CategoricalImputer
from sklearn.ensemble           import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection  import SelectKBest
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import accuracy_score, median_absolute_error, make_scorer
from sklearn.model_selection    import GridSearchCV, KFold, RandomizedSearchCV, train_test_split 
from sklearn.pipeline           import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing      import MultiLabelBinarizer, Imputer, StandardScaler, LabelEncoder

from initial_data_prep          import df, y, cols, test, test_y, TRAINING_SET_MEAN

# ______________________________________________________________________________
# Module level controls.

if not sys.warnoptions:
    warnings.simplefilter("ignore")

parser =  argparse.ArgumentParser()
parser.add_argument('--balance', action='store_true')
args = parser.parse_args()

# ______________________________________________________________________________
# Data processing functions.

def balance_data(df, y, do_undersample):
  """
  Training data is unblanaced. There are many more 0's than 1's.
  In this case, we undersample the 0 class to even out in an attempt
  to learn more about the 1 class in our model.

  If `do_undersample` set to `False`, then we just move on with 
  regular, imbalanced data.
  """
  if do_undersample:
    print('Under sampling the \'0\' class of our outcome data...')
    # Under sample -50K so we can better learn.
    ones  = df[df['binary_income']==1]
    zeros = df[df['binary_income']==0]
    
    subsampled_df = pd.concat([ones, zeros.sample(ones.shape[0])])
    subsampled_y  = subsampled_df['binary_income']
    subsampled_df = subsampled_df.drop('binary_income',axis=1)
    
    return subsampled_df, subsampled_y
  
  else:
    return df, y

def pre_MultiLabelBinarizer(vars):
  """
  Transform labels from strings to numbers. 
 
  This inelegant solution is needed as if we have a category in the test
  set that is not originally LabelEncoded, then we will not be able to 
  apply a `transform` method later on.
  """
  for var in vars:
    m = DataFrameMapper([(var, [CategoricalImputer(), LabelEncoder()])],  df_out = True)
    m.fit_transform(pd.concat([test,df]))  # Get all categorical outcomes so we have same codes for all.

    X_train[var+'_encoded'] = m.transform(X_train)
    X_test[var+'_encoded']  =  m.transform(X_test)
    test[var+'_encoded']    = m.transform(test)

  return X_train, X_test, test

# ______________________________________________________________________________
# Imputation and feature selection. 

mapper = DataFrameMapper([
                          # Categorical features
                          # ( ['reason'],        [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['migration'],     [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['fill'],          [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['live'],          [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['migrationreg'],  [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['migrationwithinreg'], [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['citizenship'],        [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['state'],              [CategoricalImputer(), LabelEncoder()] ),
                          # # ( ['migrationmsa'],     [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['mace'],               [CategoricalImputer(), LabelEncoder()] ),
                          # # ( ['enrolled'],         [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['hispanic'],           [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['countrybs'],          [CategoricalImputer(), LabelEncoder()] ),
                          # # ( ['member'],           [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['countrybf'],          [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['own'],                [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['countrybm'],          [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['veterans'],           [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['family'],             [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['full'],               [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['marital'],        [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['detailedhs'],     [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['tax'],            [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['detailedhf'],     [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['class'],          [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['majoric'],        [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['industry'],       [CategoricalImputer(), LabelEncoder()] ),
                          # ( ['majoroc'],        [CategoricalImputer(), LabelEncoder()] ),
                          ( ['sex'],              [CategoricalImputer(), LabelEncoder()] ),
                          ( ['education'],        [CategoricalImputer(), LabelEncoder()] ),

                          # Categorical features, represented as numeric
                          # ( ['year'],              [Imputer(strategy='median')] ),

                          # Categorical features previously encoded
                          ( ['occupation_encoded'],        [CategoricalImputer()] ),
                          # ( ['industry_encoded'],              [Imputer(strategy='median')] ),
                          # ( ['detailedhf_encoded'],              [Imputer(strategy='median')] ),

                          # Numeric features
                          ( ['age'],              [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['wage'],             [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['weeks'],            [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['capital_gains'],    [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['dividends'],        [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['num'],              [Imputer(strategy='median'), StandardScaler()] ),
                          # ( ['capital_losses'],   [Imputer(strategy='median'), StandardScaler()] ),
                          # ( ['instance'],         [Imputer(strategy='median'), StandardScaler()] ),

                          # My created features
                          ( ['ttl_clcltd_ernings'],  [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['white'],               [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['professional'],        [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['ad_degree'],           [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['younger22'],           [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['older65'],             [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['jointu65_tax'],        [Imputer(strategy='median'), StandardScaler()] ),
                          ( ['occupation2'],         [Imputer(strategy='median'), StandardScaler()] )

              			     ],  
                         df_out = True)

def fit_andor_transform(X_train, X_test):
  """
  Pass training data (`X_train`) through fit_transform.

  Pass `X_test` just through transform to ensure no 
  data leakage.
  """
  fit_transformed_df = mapper.fit_transform(X_train)

  X_np      = np.asarray(fit_transformed_df)
  X_np_test = np.asarray(mapper.transform(X_test))

  return X_np, X_np_test

# ______________________________________________________________________________
# Sklearn pipeline instantiation

def build_pipe():
  pipeline = Pipeline([
            ('SelectKBest', SelectKBest()),
            ('GradientBoostingClassifier', GradientBoostingClassifier(verbose = 1))
            # Other model types that we considered.
            # ('LogisticRegression', LogisticRegression())
            # ('RandomForestClassifier', RandomForestClassifier(verbose = 1))
                      ])

  # Possible values for GridSearchCV.
  param_dist = {'SelectKBest__k': ['all'],
                'GradientBoostingClassifier__n_estimators': [40],
                'GradientBoostingClassifier__max_depth': [5],
                'GradientBoostingClassifier__subsample': [.7]}

  pipe = GridSearchCV(estimator = pipeline, 
                      param_grid=param_dist, 
                      cv=KFold(n_splits=4, random_state = 42),
                      n_jobs=-1, # can set to -1 (num machine cores) if not using custom scorers.
                      verbose=1)
  return pipe

def print_model_info(do_print):
  if do_print:
    print('\n Best parameters:')
    print(pipe.best_params_)

    print('\n Feature importances:')
    xs = mapper.fit_transform(X_train).columns[pipe.best_estimator_.named_steps['SelectKBest'].get_support()]
    varimp = pipe.best_estimator_.named_steps['GradientBoostingClassifier'].feature_importances_
    varimp_df = pd.DataFrame({'Cols':xs, 'Importances': varimp})
    print(varimp_df.sort_values(by = 'Importances', ascending= False))

    print('Learning data train and test error:')
    print('Train set accuracy_score is {0:.3f}'.format(accuracy_score(pipe.predict(X_np), y_train)))
    print('Test set accuracy_score is {0:.3f}'.format(accuracy_score(pipe.predict(X_np), y_test)))

# ______________________________________________________________________________
# Fit model on census_income_test.csv

def get_fully_held_out_preds(test, mapper, pipe):
  """
  Apply transformation to fully held out test data,
  and then make predictions.

  `predict_proba` returns array of class probabilities,
  which is good for ensembling later on.
  """
  test_np_full = np.asarray(mapper.transform(test))
  preds        = pipe.predict_proba(test_np_full)

  return preds

if __name__ == '__main__':
  df, y                            = balance_data(df, y, do_undersample = args.balance)
  X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
  X_train, X_test, test            = pre_MultiLabelBinarizer(['industry','detailedhf', 'occupation'])
  X_np, X_np_test                  = fit_andor_transform(X_train, X_test)
  pipe                             = build_pipe()

  pipe.fit(X_np, y_train.values.reshape(-1,))
  
  fully_held_out_test_preds        = get_fully_held_out_preds(test, mapper, pipe)
  if args.balance:
    with open('balanced_preds.pkl', 'wb') as f:
      pickle.dump(fully_held_out_test_preds, f)
  else:
    with open('test_preds.pkl', 'wb') as f:
      pickle.dump(fully_held_out_test_preds, f)



