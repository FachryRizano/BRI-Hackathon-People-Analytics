import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from normalize_data import normalize_data

df_test = pd.read_csv("test.csv")
df = pd.read_csv("train.csv")

df_test = normalize_data(df_test,False)

X = df_test.drop('achievement_target_3',axis=1)
y = df_test['achievement_target_3']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200)
# model = DecisionTreeClassifier()
model.fit(X_train,y_train)
# pred = model.predict(X_test)
# print(accuracy_score(y_test,pred))
# print(classification_report(y_test,pred))
# print(confusion_matrix(y_test,pred))

df = normalize_data(df,False)
# print(df['achievement_target_3'])
for i in enumerate(df):
    
print(df.drop('achievement_target_3',axis=1).loc[i])
# df['achievement_target_3'].fillna(model.predict()
# print(df['achievement_target_3'])
# print(df.head())
# print(df.info())