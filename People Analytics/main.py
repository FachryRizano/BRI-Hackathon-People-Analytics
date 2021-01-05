import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score,f1_score,roc_curve,auc,confusion_matrix,classification_report,plot_roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from normalize_data import normalize_data
from sklearn.naive_bayes import GaussianNB
from bayes_opt import BayesianOptimization
# from sklearn.impute import KNNImputer,SimpleImputer
# from sklearn.experimental import enable_iterative_imputer
from lightgbm import LGBMClassifier
# from sklearn.impute import IterativeImputer
# def plot_roc_curve(fpr, tpr):
#     plt.plot(fpr, tpr, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()

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

        lgbm_model = LGBMClassifier(**params)
        lgbm_model.fit(self.x_train, self.y_train)
        y_pred = lgbm_model.predict_proba(self.x_test)
        predictions = y_pred[:,1]
#         rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
#         return -1*rmse
        acc = roc_auc_score(self.y_test,predictions)
        return acc

df_test= pd.read_csv("test.csv")
df = pd.read_csv("train.csv") #dataset untuk ditrain, include label
print(df.isnull().sum())
# mice_avg_achievement = 
#coba-coba model

# KNN Imputer
# define imputer
# imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
# imputer.fit(df)
# split into input and output elements
df = normalize_data(df)
print(df.info())
print(df.head())
print(df.isnull().sum())
# imputer = IterativeImputer()
# # imputer.fit(X_train)
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)

# merubah object menjadi category column
obj_columns = df.select_dtypes(['object']).columns
df[obj_columns] = df[obj_columns].astype('category')
#merubah category menjadi int column
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x:x.cat.codes)
#menaruh kolum Best Performance ke index terakhir 
col = df.pop("Best Performance")
df.insert(26,col.name,col)

print(df.head())
print(df.isnull().sum())
print(df.info())
# print(df.info())
    
# df = df.fillna(df.mean())
# print(df.corr()['Best Performance'][:-1].sort_values().plot(kind='bar'))
# # plt.show()
data = df.values
ix = [i for i in range(data.shape[1]) if i != 26]
X, y = data[:, ix], data[:, 26]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=42)
scaler = MinMaxScaler()
X= scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

# print(X.shape)
# print(y.shape)



# model = LogisticRegression()
# model.fit(X_train,y_train)
# rf_probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
# rf_probs = rf_probs[:, 1]
# auc = roc_auc_score(y_test,rf_probs)
# print('AUC: %.2f ' % auc)
# print(roc_auc_score(y_test,rf_probs))
# fpr,tpr,thresholds = roc_curve(y_test,rf_probs)
# plot_roc_curve(fpr,tpr)

#LGBM model
params = {'min_child_weight': 0.6715,
                                            'max_depth': 36,
                                            'num_leaves': 117,
                                            'min_child_samples' :24,
                                            'bagging_fraction' : 0.8538,
                                            'lambda_l1' : 0.7467,
                                            'lambda_l2' : 0.6911,
                                            'learning_rate':0.9107958117897483,
                                            'boosting_type':'dart',
                                            'objective':'binary',
                                            'metric':'roc_auc',
                                            'sub_feature':0.04374945990219048,
                                            'min_data':93
                                           }


from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10)
# rf = RandomForestClassifier(n_estimators=500)
# rf = GaussianNB()
rf = LGBMClassifier(**params)
# rf=DecisionTreeClassifier()
# rf = LogisticRegression()
# rf = KNeighborsClassifier(n_neighbors=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    # rf.fit(X[train], y[train])
    # lt = lgbm_target(X[train], y[train], X[test], y[test])
    rf.fit(X[train],y[train])
    viz = plot_roc_curve(rf, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()

# df_test = normalize_data(df_test)

# obj_columns = df_test.select_dtypes(['object']).columns
# df_test[obj_columns] = df_test[obj_columns].astype('category')
# #merubah category menjadi int column
# cat_columns = df_test.select_dtypes(['category']).columns
# df_test[cat_columns] = df_test[cat_columns].apply(lambda x:x.cat.codes)

# df_test = df_test.values
# df_test = scaler.transform(df_test)
# result = rf.predict_proba(df_test)
# result = result[:,1]
# pd.DataFrame(result,columns=['Best Performance']).to_csv('result_cv.csv')

# skf.get_n_splits(X,y)
# for train_index, test_index in skf.split(X,y):
#     print("Train:",train_index, "Validation:",test_index)
#     X_train, X_test= X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#     rf.fit(X_train,y_train)
#     pred = rf.predict_probs(X_test)
#     pred = pred[:,1]
#     score = roc_auc_score(y_test,pred)
#     roc.append(score)

# print("roc_score:",np.array(roc).mean())

# print(cross_val_score(rf, X, y, scoring="roc_auc", cv = 10))
# mean_score = cross_val_score(rf, X, y, scoring="roc_auc", cv = 10).mean()
# std_score = cross_val_score(rf, X, y, scoring="roc_auc", cv = 10).std()
# print(mean_score)
# print(std_score)

# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
# plt.title('Error Rate vs K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()