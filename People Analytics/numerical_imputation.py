import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
df[['Last_achievement','GPA','Avg_achievement','job_duration_as_permanent_worker']] = dataset[['Last_achievement','GPA','Avg_achievement','job_duration_as_permanent_worker']].replace(0, nan)
# retrieve the numpy array
values = df.values
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
transformed_values = imputer.fit_transform(values)
# count the number of NaN values in each column
print('Missing: %d' % isnan(transformed_values).sum())
print(sns.heatmap(df.isnull(),cmap='viridis',))
#Last Achievement %
#Regression Imputation dengan 
#numerical value
# print(df.corr()['Last_achievement_%'])
# print(sns.distplot(df['Last_achievement_%']))
# plt.show()

#GPA
#Regression Imputation
#numerical value
# print(df['GPA'].head())
# print(sns.histplot(df['GPA']))
# plt.show()

#Avg achievement %
#Regression Imputation
#numerical value
# print(df.columns)
# print(sns.histplot(df['Avg_achievement_%']))
# plt.show()

# job_duration_as_permanent_worker %
#high correlation dengan job_duration_from_training
#Regression Imputation
# job_duration_as_permanent_worker : lama bekerja sebagai pekerja tetap
#float (numeriical)
# print(df.info()['job_duration_as_permanent_worker'])
# print(sns.countplot(df['job_duration_as_permanent_worker']))
# plt.show()