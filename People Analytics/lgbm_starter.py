import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from bayes_opt import BayesianOptimization

all_train_data = pd.read_csv('train.csv')
all_test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
dropped_columns = ['year_graduated']
y_train = all_train_data['Best Performance']
X_train = all_train_data.drop('Best Performance',axis=1)
for col in dropped_columns:
    X_train = X_train.drop(col,axis=1)
X_test = all_test_data
for col in dropped_columns:
    X_test = X_test.drop(col,axis=1)
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=0.2, random_state=38)
cat_columns = ['job_level', 'person_level', 'Employee_type', 'Employee_status', 'gender', 'marital_status_maried(Y/N)', 'Education_level', 'achievement_target_1', 'achievement_target_2', 'achievement_target_3']
for cat in cat_columns:
    labelencoder = LabelEncoder()
    X_train_[cat] = labelencoder.fit_transform(X_train_[cat].astype(str))
    X_test_[cat] = labelencoder.transform(X_test_[cat].astype(str))
for cat in cat_columns:
    labelencoder = LabelEncoder()
    X_train[cat] = labelencoder.fit_transform(X_train[cat].astype(str))
    X_test[cat] = labelencoder.transform(X_test[cat].astype(str))        
params = {'min_child_weight': 0.6715,
                                              'max_depth': 12,
                                              'num_leaves': 20,
                                            'min_child_samples' :24,
                                            'bagging_fraction' : 0.8538,
                                            'lambda_l1' : 0.7467,
                                            'lambda_l2' : 0.6911
                                           }
model_lgb = lgb.LGBMClassifier(**params)
model_lgb.fit(X_train,y_train)
model_lgb.predict_proba(X_test)[:,1]
sample_submission['Best Performance'] = model_lgb.predict_proba(X_test)[:,1]
sample_submission.to_csv('submission.csv')
class lgbm_target :
    def __init__(self, x_train, y_train, x_test, y_test) :
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def clean_param(self, param) :
        params = {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0, 
                  'importance_type':'split', 'learning_rate':0.1,
                  'min_child_samples':20, 'min_split_gain':0.0, 'n_estimators':100, 'objective':None,
                  'random_state':0, 'reg_alpha':0.0, 'reg_lambda':0.0, 'silent':True,
                  'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0}
        params['num_leaves'] = int(param['num_leaves'])
        params['min_child_weight'] = int(param['min_child_weight'])
        params['max_depth'] = int(param['max_depth'])
        params['learning_rate'] = 0.1
        params['min_data_in_bin'] = 1
        params['min_data'] = 1
        
        params['min_child_samples'] = int(param['min_child_samples'])
        params['bagging_fraction'] = param['bagging_fraction']
        params['lambda_l1'] = param['lambda_l1']
        params['lambda_l2'] = param['lambda_l2']

        return params
        
    def evaluate(self, min_child_weight, max_depth, num_leaves,
                min_child_samples, bagging_fraction, lambda_l1, lambda_l2):
        params = {'num_leaves':num_leaves, 
                  'min_child_weight':min_child_weight, 
                  'max_depth':max_depth,
                 'min_child_samples':min_child_samples,
                 'bagging_fraction' : bagging_fraction,
                 'lambda_l1' : lambda_l1,
                 'lambda_l2' : lambda_l2}
        
        params = self.clean_param(params)

        lgbm_model = lgb.LGBMClassifier(**params)
        lgbm_model.fit(self.x_train, self.y_train)
        y_pred = lgbm_model.predict_proba(self.x_test)
        predictions = y_pred[:,1]
#         rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
#         return -1*rmse
        acc = roc_auc_score(self.y_test,predictions)
        return acc

lt = lgbm_target(X_train_, y_train_, X_test_, y_test_)
lgbmBO = BayesianOptimization(lt.evaluate, {'min_child_weight': (0.01, 1),
                                              'max_depth': (7, 15),
                                              'num_leaves': (5, 50),
                                            'min_child_samples' :(10,50),
                                            'bagging_fraction' : (0.5,1),
                                            'lambda_l1' : (0,1),
                                            'lambda_l2' : (0,1)
                                           }, 
                             random_state=3)

lgbmBO.maximize(init_points=30, n_iter=20)
y_label_adv = np.zeros((X_train.shape[0]+ X_test.shape[0]))
y_label_adv[:X_train.shape[0]] = 1
adversarial_data = pd.concat((X_train,X_test))
model_lgb = lgb.LGBMClassifier()
X_train_, X_test_, y_train_, y_test_ = train_test_split(adversarial_data, y_label_adv, test_size=0.33, random_state=38)

model_lgb.fit(X_train_,y_train_)
y_pred  = model_lgb.predict(X_test_)
roc_auc_score(y_pred,y_test_)