# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:23:57 2019

@author: Woody
"""
##在自己划分的测试集上的分数：0.761，提交后的分数0.731

import pandas as pd  
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

##导入测试集，训练集，和所有训练集
df1 = pd.read_csv('train.csv',index_col=False)
df2 = pd.read_csv('train_label.csv',index_col=False)
df3 = pd.merge(df1,df2,on = 'ID')
df4 = pd.read_csv('test.csv')
ID1 = df3['ID']
ID2 = df4['ID']

def parsers(line):
    x = int(line[8]+line[9])
    y = int(line[11]+line[12])
    return x+0.01*y

feature = list(df3.columns)
feature.remove("label")
features = feature.copy()
features.remove('date')
del feature[1]
del feature[0]
features.remove('ID')
weekdaycols = ['weekday_' + str(i) for i in range(0,7)]

def preprocess1(df3):
    a = pd.to_datetime(df3['date'])
    df3['weekday'] =a.apply(lambda x:x.weekday()) 
    df3['date'] = df3['date'].apply(parsers)
    df3['hour'] = df3['date'].apply(lambda x:(x-int(x))*100)
    tmpdf = pd.get_dummies(df3['weekday'])
    tmpdf.columns = weekdaycols
    df3[weekdaycols] = tmpdf
    return df3

df3 = preprocess1(df3)
df4 = preprocess1(df4)

add_features1 = ['hour']+weekdaycols

remove_features = ['E21','E3','E16','E13']

for i in add_features1:
    features.append(i)
for i in remove_features:
    features.remove(i)

mean_B1 = df3.groupby('B1')['label'].mean()
mean_B2 = df3.groupby('B2')['label'].mean()
mean_B3 = df3.groupby('B3')['label'].mean()
mean_A1 = df3.groupby('A1')['label'].mean()
mean_A2 = df3.groupby('A2')['label'].mean()
mean_A3 = df3.groupby('A3')['label'].mean()
count_B2 = df3.groupby('B2')['label'].count()
count_B3 = df3.groupby('B3')['label'].count()
count_C1 = df3.groupby('C1')['label'].count()
count_C2 = df3.groupby('C2')['label'].count()
count_C3 = df3.groupby('C3')['label'].count()
count_D1 = df3.groupby('D1')['label'].count()
count_hour = df3.groupby('hour')['label'].count()
count_B1_B2_B3 = df3.groupby(['B1','B2','B3'])['label'].count()
count_B1_B2_B3_index = list(count_B1_B2_B3.index )

df3['mean_A2'] = df3['A2'].apply(lambda x:mean_A2[x])
df3['mean_A3'] = df3['A3'].apply(lambda x:mean_A3[x])
df3['mean_B1'] = df3['B1'].apply(lambda x:mean_B1[x])
df3['mean_B2'] = df3['B2'].apply(lambda x:mean_B2[x])
df3['mean_B3'] = df3['B3'].apply(lambda x:mean_B3[x])
df3['count_B2'] = df3['B2'].apply(lambda x:count_B2[x])
df3['count_B3'] = df3['B3'].apply(lambda x:count_B3[x])
df3['count_C1'] = df3['C1'].apply(lambda x:count_C1[x])
df3['count_C2'] = df3['C2'].apply(lambda x:count_C2[x])
df3['count_C3'] = df3['C3'].apply(lambda x:count_C3[x])
df3['count_D1'] = df3['D1'].apply(lambda x:count_D1[x])
df3['count_hour'] = df3['hour'].apply(lambda x:count_hour[x])

df4['mean_A1'] = df4['A1'].apply(lambda x:mean_A1[x] if x in mean_A1.index else 0.)
df4['mean_A2'] = df4['A2'].apply(lambda x:mean_A2[x] if x in mean_A2.index else 0.1)
df4['mean_A3'] = df4['A3'].apply(lambda x:mean_A3[x] if x in mean_A3.index else 0.1)
df4['mean_B1'] = df4['B1'].apply(lambda x:mean_B1[x] if x in mean_B1.index else 0.1)
df4['mean_B2'] = df4['B2'].apply(lambda x:mean_B2[x] if x in mean_B2.index else 0.1)
df4['mean_B3'] = df4['B3'].apply(lambda x:mean_B3[x] if x in mean_B3.index else 0.1)
df4['count_B2'] = df4['B2'].apply(lambda x:count_B2[x] if x in count_B2.index else count_B2.mean())
df4['count_B3'] = df4['B3'].apply(lambda x:count_B3[x] if x in count_B3.index else count_B3.mean())
df4['count_C1'] = df4['C1'].apply(lambda x:count_C1[x] if x in count_C1.index else count_C1.mean())
df4['count_C2'] = df4['C2'].apply(lambda x:count_C2[x] if x in count_C2.index else count_C2.mean())
df4['count_C3'] = df4['C3'].apply(lambda x:count_C3[x] if x in count_C3.index else count_C3.mean())
df4['count_D1'] = df4['D1'].apply(lambda x:count_D1[x] if x in count_D1.index else count_D1.mean())
df4['count_hour'] = df4['hour'].apply(lambda x:count_hour[x] if x in count_hour.index else count_hour.mean())

add_features2 = ['mean_A1','mean_A2','mean_A3','mean_B1','mean_B2','mean_B3'\
                 ,'count_B2','count_B3','count_C1','count_C2','count_C3','count_D1','count_hour','count_B1_B2_B3']
for i in add_features2:
    features.append(i)

label = df3['label']
df3 = df3[features]
df3.insert(58,'label',label)
df4 = df4[features]
df4['label'] = '$'
df_all = pd.concat([df3,df4])

features_ = features.copy()
features__ = []
for i in weekdaycols:
    features_.remove(i)
for i in add_features2:
    features_.remove(i)
    
#10 0.7590   20 0.7591   30 0.7586   说明20-30之间有的特征有害为B1的各类
#并且40-50之间也有有害特征hour
#70的时候就达到了0.7599
#400的时候达到0.759088
#加入99的B3基本不提升,加入398的E1也不提升,加入1288的b3也不提升,1326的A3有害
for i in features_:
    if len(df_all[i].unique())<160 and i not in ['B1','hour','B3','E1','B2','A3','A1','E2']:
        features__.append(i)

for i in features__:
    dummies = pd.get_dummies(df_all[i],prefix = i)
    df_all[dummies.columns] = dummies
    features = features+list(dummies.columns)

df3 = df_all[df_all['label']!='$']
df4 = df_all[df_all['label'] == '$']
features = list(df3.columns)
features.remove('label')

model1 = CatBoostRegressor(max_depth = 4)
x_train, x_test, y_train, y_test = train_test_split(df3[features],df3['label'], test_size = 0.3,random_state=44)
y_train = y_train.astype('int')  
y_test = y_test.astype('int')
model1.fit(x_train,y_train)
y_pred1_1 = model1.predict(x_test)
y_pred1 = y_pred1_1
score2 = roc_auc_score(y_test,y_pred1)

model1.fit(df3[features],df3['label'])
print("{}测试集上auc为:{}".format(model1,score2))

y_pred2 = model1.predict(df4[features])
y_pred2 = np.array(y_pred2)
df5 = pd.DataFrame()
df5['ID'] = ID2
df5['label'] = y_pred2
df5['label'] = df5['label'].apply(lambda x:0 if x<0 else x)
df5['label'] = df5['label'].apply(lambda x :1 if x>1 else x)
df5.to_csv('submission.csv',index=0)












