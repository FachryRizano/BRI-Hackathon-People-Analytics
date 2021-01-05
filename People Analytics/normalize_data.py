import pandas as pd
import numpy as np
def normalize_data(df):
    #hanya 12 data yg null untuk column Employee_Type
    
    # df['Employee_type']= df['Employee_type'].fillna(df['Employee_type'].mode().iloc[0])
    df= df[df['Employee_type'].notna()]

    #number_of_dependences drop karena sudah ada yg untuk female dan male
    #achievement_target_1 bisa dibuang karena dia memiliki ratio yang tidak signifikan best performance dengan tidak
    #achievement_target_2 bisa dibuang karena dia memiliki ratio yang tidak signifikan best performance dengan tidak
    #achievement_target_3 bisa dibuang karena dia memiliki ratio yang tidak signifikan antara best performance dengan tidak
    #year_graduated drop karena isinya gak jelas
    # Achievement_above_100%_during3quartal karena memiliki ratio 
    #GPA diisi dengan mean dari column GPA
    #Avg achievement % diisi dengan rata2 
    #Last achievement % diisi dengan rata2
    df = df.drop(['number_of_dependences (male)','number_of_dependences (female)','year_graduated','Employee_status'],axis=1)
    # df['year_graduated']=df['year_graduated'].replace('.','2013') #most frequent
    # df['year_graduated']=df['year_graduated'].replace('0','2013') #most frequent
    # df['year_graduated']=df['year_graduated'].replace('1016','2016') #mungkin 2016
    # df['year_graduated']=df['year_graduated'].replace('207','2007') #mungkin 2007
    # df['year_graduated']=df['year_graduated'].replace('209','2009') #mungkin 2009
    # df['year_graduated']=df['year_graduated'].replace('2201','2001') #mungkin 2001
    # df['year_graduated']=df['year_graduated'].replace('2999','2000') #mungkin 2000
    # df['year_graduated']=df['year_graduated'].replace('3.05','2013') #most frequent
    # df['year_graduated']=df['year_graduated'].replace('3.18','2013') # most frequent
    # df['year_graduated']=df['year_graduated'].replace('3013','2013') #mungkin 2013
    # df['year_graduated']=df['year_graduated'].replace('9999','1999') #mungkin 1999
    # df['year_graduated']=df['year_graduated'].replace('\\N','2013') #most frequent
    # df = df[
    #     (df.year_graduated != ".") & 
    #     (df.year_graduated != "0") & 
    #     (df.year_graduated != "1016") & 
    #     (df.year_graduated !="207") &
    #     (df.year_graduated !="209") &
    #     (df.year_graduated !="2201") &
    #     (df.year_graduated !="2999") &
    #     (df.year_graduated !="3.05") &
    #     (df.year_graduated !="3.18") &
    #     (df.year_graduated !="3013") &
    #     (df.year_graduated !="9999") &
    #     (df.year_graduated !="\\N")]
    # df = df.drop(['number_of_dependences (male)','number_of_dependences (female)'],axis=1)
    
    df['Achievement_above_100%_during3quartal'] = df['Achievement_above_100%_during3quartal'].fillna(df['Achievement_above_100%_during3quartal'].mean())
    df['Avg_achievement_%'] = df['Avg_achievement_%'].fillna(df['Avg_achievement_%'].mean())
    df['Last_achievement_%'] = df['Last_achievement_%'].fillna(df['Last_achievement_%'].mean())
    df['Education_level'] = df['Education_level'].fillna(df['Education_level'].mode().iloc[0])
    df['age'] = 2020 - df['age']
    df.drop_duplicates(inplace=True)
    df['GPA'] = df['GPA'].fillna(df['GPA'].mean())
    df['job_duration_as_permanent_worker'] = df['job_duration_as_permanent_worker'].fillna(df['job_duration_as_permanent_worker'].mean())
    # df = df.fillna(df.mean())
    df['achievement_target_3'] = df['achievement_target_3'].replace('not_reached','not reached')
    df['achievement_target_2'] = df['achievement_target_2'].replace('Pencapaian < 50%','achiev_< 50%')
    df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
    df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
    df['achievement_target_2'] = df['achievement_target_2'].replace("Pencapaian > 1.5",'achiev_> 1.5')
    df['achievement_target_1'] = df['achievement_target_1'].replace('Pencapaian < 50%','achiev_< 50%')
    df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 50%-100%",'achiev_50%-100%')
    df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian 100%-150%",'achiev_100%-150%')
    df['achievement_target_1'] = df['achievement_target_1'].replace("Pencapaian > 1.5",'achiev_> 1.5')
    # df = df[df['Employee_status']!= "Contract"]
    df['achievement_target_1'] = df['achievement_target_1'].fillna(df['achievement_target_1'].mode().iloc[0])
    df['achievement_target_2'] = df['achievement_target_2'].fillna(df['achievement_target_2'].mode().iloc[0])
    df['achievement_target_3'] = df['achievement_target_3'].fillna(df['achievement_target_3'].mode().iloc[0])
    df = df[df['achievement_target_3'] != "NONE"]
    # df['year_graduated'] = df['year_graduated'].fillna(df['year_graduated'].mode().iloc[0])
    return df