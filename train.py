# necessary Imports
#import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import optuna
from sklearn.preprocessing import LabelEncoder
#from sklearn.datasets import load_boston
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor

# loading the data
train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test' , skiprows = 1, header = None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', \
'marital_status', 'occupation','relationship', 'race', 'sex', 'capital_gain',\
'capital_loss', 'hours_per_week', 'native_country', 'wage_class']

train_set.columns = col_labels
test_set.columns = col_labels

# To use Categorical features as is for XGboost, need to makr thier type as "Category"
# ref https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html
train_set["workclass"]=train_set["workclass"].astype("category")
train_set["education"]=train_set["education"].astype("category")
train_set["marital_status"]=train_set["marital_status"].astype("category")
train_set["occupation"]=train_set["occupation"].astype("category")
train_set["relationship"]=train_set["relationship"].astype("category")
train_set["race"]=train_set["race"].astype("category")
train_set["sex"]=train_set["sex"].astype("category")
train_set["native_country"]=train_set["native_country"].astype("category")

test_set["workclass"]=test_set["workclass"].astype("category")
test_set["education"]=test_set["education"].astype("category")
test_set["marital_status"]=test_set["marital_status"].astype("category")
test_set["occupation"]=test_set["occupation"].astype("category")
test_set["relationship"]=test_set["relationship"].astype("category")
test_set["race"]=test_set["race"].astype("category")
test_set["sex"]=test_set["sex"].astype("category")
test_set["native_country"]=test_set["native_country"].astype("category")

# splitting the features and label from "Training" dataset
x_train = train_set.drop(columns='wage_class')
y_train = train_set['wage_class']

# splitting the features and label from "Testing" dataset
x_test=test_set.drop(columns='wage_class')
y_test=test_set['wage_class']

# As the label has Categorical/String values as '<=50K' or ' >50K' , converting them into 0 or 1 (Binary Classification).
label_encoder = LabelEncoder()
y_train_en = label_encoder.fit_transform(y_train)  # for Training label
y_test_en = label_encoder.fit_transform(y_test)   # for Testing label

# Creating the Model -->
# Here instead of performing encoding of categorical columns in featureset(x),
# we have used "enable_categorical=True" parameter of XGboost classifier.
# Reference- https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html
# This saved lot of time, effort and extra columns that would need to be added in feature set because of encoding
xgb_default_model=xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True,use_label_encoder=False)


# fitting the model
xgb_default_model.fit(x_train,y_train_en)
# Here the parameters of this model are as below:
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=True,
#               gamma=0, gpu_id=0, importance_type=None,
#               interaction_constraints='', learning_rate=0.300000012,
#               max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
#               monotone_constraints='()', n_estimators=100, n_jobs=8,
#               num_parallel_tree=1, predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='gpu_hist', use_label_encoder=False,
#               validate_parameters=1, verbosity=None)

print("Training Score = ", xgb_default_model.score(x_train,y_train_en))
print("Testing Score = ", xgb_default_model.score(x_test,y_test_en))

# saving the model to the local file system
filename = 'XGB_clfModel_earning.pickle'
pickle.dump(xgb_default_model,open(filename,'wb'))

# prediction using the saved model
Loaded_model=pickle.load(open(filename,'rb'))

print("Train.py ran successfully. Model is built and loaded")

#for i in range(0,500,50):
 #   prediction_output = loaded_model.predict(scaler.transform([x.iloc[i]]))
  #  print("\n *** For input", np.array(x.iloc[i],dtype=str),"\n prediction_output = ",\
   #       prediction_output, "\n Expected output= ", y[i])


# *** HyperParameter Tuning using Optuna ***
# Lets See if using Optuna we can find optimized parameters which will give better accuracy

# def objective(trail, train_x= x_train, train_y= y_train_en, test_x= x_test, test_y= y_test_en):
#     #train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.10,random_state=30)
#     params={
#         'enable_categorical':True,
#         'use_label_encoder':False,
#         'tree_method':'gpu_hist',
#         'n_estimator': [500,1000],
#         'max_depth' : trail.suggest_categorical('max_depth',[4,5,6,7,8]),
#         'learning_rate' :trail.suggest_categorical('learning_rate',[0.1,0.2,0.300000012,0.4])
#         }
#     xgb_clf_model=xgb.XGBClassifier(**params)
#     xgb_clf_model.fit(train_x,train_y,eval_set=[(test_x,test_y)],verbose=True)
#     #pred_xgb=xgb_clf_model.predict(test_x)
#     #rmse=mean_squared_error(test_y,pred_xgb)
#     #training_score=xgb_clf_model.score(train_x,train_y)
#     testing_score=xgb_clf_model.score(test_x,test_y)
#     return testing_score #,training_score,;

# opt_param=optuna.create_study(direction='minimize')
# opt_param.optimize(objective,n_trials=10)
#
# best_param=opt_param.best_trial.params
# print("Best Parameters found by Optuna: ",best_param)
#
# opt_param.trials_dataframe()
#
# xgb_opt_model=xgb.XGBClassifier(**best_param,enable_categorical=True,use_label_encoder=False,tree_method='gpu_hist')
# #here ** because best_param is the dictionary with key:value pair
#
# xgb_opt_model.fit(x_train,y_train_en)
# '''XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, enable_categorical=True,
#               gamma=0, gpu_id=0, importance_type=None,
#               interaction_constraints='', learning_rate=0.4, max_delta_step=0,
#               max_depth=8, min_child_weight=1, missing=nan,
#               monotone_constraints='()', n_estimators=100, n_jobs=8,
#               num_parallel_tree=1, predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#               tree_method='gpu_hist', use_label_encoder=False,
#               validate_parameters=1, verbosity=None)'''

# xgb_opt_model.score(x_train,y_train_en)
# xgb_opt_model.score(x_test,y_test_en)

# Even after multiple trials of Parameter turning with Optuna,
# the default model with learning_rate=0.300000012,max_depth=6 is giving most stable model
# with Train score= 90.04% and Testing Score= 87.32 %
# Hence finally I am using the Model created with default parameters
# Hence all the code related to Optuna is commented above.