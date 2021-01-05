import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
df = pd.read_csv("train.csv")
#ACHIEVMENT_TARGET_3
#solve dengan membuat model KNN
df['achievement_target_3'] = df['achievement_target_3'].replace('not_reached','not reached')
# print(sns.countplot(x='Best Performance', hue='achievement_target_3',data=df))

#rubah menjadi dummy variable agar bisa dibaca correlationnya
dummy_at_3 = pd.get_dummies(df['achievement_target_3'],dummy_na=True,drop_first=True)
# dummy_at_3['nan_3'] = dummy_at_3['nan']
# dummy_at_3 = dummy_at_3.drop('NaN',axis=1)
dummy_at_3['nan_3'] = dummy_at_3.iloc[:,1]
dummy_at_3 = dummy_at_3[['reached','nan_3']]
# print(dummy_at_3.head())
#mengisi nilai null dengan sesuatu

#ACHIEVEMENT_TARGET_2
#KNN
#Pencapaian < ... = achiev_ ...
df['achievement_target_2'] = df['achievement_target_2'].replace('Pencapaian < 50%','achiev_< 50%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian > 1.5",'achiev_> 1.5')
#rubah menjadi dummy variable beserta denga column yang bernilai null
dummy_at_2 = pd.get_dummies(df['achievement_target_2'],dummy_na=True,drop_first=True)
dummy_at_2['nan_2'] = dummy_at_2.iloc[:,4]
dummy_at_2 = dummy_at_2[['achiev_100%-150%','achiev_50%-100%','achiev_< 50%','achiev_> 1.5','nan_2']]
# print(dummy_at_2.info())

#mengisi nilai null dengan sesuatu
# print(dummy_at_2)

#ACHIEVEMENT_TARGET_1
#KNN
df['achievement_target_1'] = df['achievement_target_1'].replace('Pencapaian < 50%','achiev_< 50%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian > 1.5",'achiev_> 1.5')

dummy_at_1 = pd.get_dummies(df['achievement_target_1'],dummy_na=True,drop_first=True)
dummy_at_1['nan_1'] = dummy_at_1.iloc[:,4]
dummy_at_1 = dummy_at_1[['achiev_100%-150%','achiev_50%-100%','achiev_< 50%','achiev_> 1.5','nan_1']]

# df = pd.concat([df.drop(['achievement_target_1','achievement_target_2','achievement_target_3'],axis=1),dummy_at_3],axis=1)
# print(sns.heatmap(df[:][-1:].corr(),yticklabels=False,cmap="viridis",cbar=False,annot=True))
# plt.show()

# print(df.corr()['Last_achievement_%'])



#Achievement_above_100%_during3quartal
#udah rapih
#categorical 0-1-2-3
# print(sns.countplot(x='Achievement_above_100%_during3quartal',data=df))
# plt.show()

#Education level
#categorical value
print(sns.countplot(x='Education_level',data=df))
plt.show()


#Year Graduated
# year_graduated : Tahun lulus
#Tahun bentuknya object perlu di convert jadi numerical value
# print(df.info())
# print(sns.countplot(df['year_graduated']))
# plt.show()



#Employee_status
# Employee_status : Status Pekerja (tetap/kontrak)
#perlu dirubah menjadi dummy variable
#Cateogrical (Permanent dan Contract)
# print(sns.countplot(df['Employee_status']))
# plt.show()

#Fungsi KNN
encoder = OrdinalEncoder()
imputer = KNN()

# create a list of categorical columns to iterate over
cat_cols = ['achievement_target_1','achievement_target_2','achievement_target_3','year_graduated','Education_level','marital_status_maried(Y/N)','gender','Employee_status','Employee_type','person_level','job_level']

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

# #create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(df[columns])