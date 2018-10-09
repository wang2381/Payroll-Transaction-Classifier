import pandas as pd
import numpy as np
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

Location=input('Please enter the path for bank transactions CSV file\n\n')
try:

	df_trans = pd.read_csv(Location,header=None,low_memory=False,skiprows=1,usecols=[0,1,2,3,4],names=['id','transid','amount','desc','date'])

except:
	print('Cannot open',Location,'!')
	quit()

df_trans_missing1 = df_trans[pd.isnull(df_trans['id'])]
df_trans_missing2 = df_trans[pd.isnull(df_trans['transid'])]
df_trans_missing3 = df_trans[pd.isnull(df_trans['amount'])]
df_trans_missing4 = df_trans[pd.isnull(df_trans['desc'])]
df_trans_missing5 = df_trans[pd.isnull(df_trans['date'])]
df_trans_missing=pd.concat([df_trans_missing1,df_trans_missing2,df_trans_missing3,df_trans_missing4,df_trans_missing5])

print('There are', len(df_trans_missing), 'records missing. Check Missing.csv for missing values!\n\n')
print('Processing continues...\n\n')
print('Dropping NULL records...\n\n')


df_nm=df_trans.copy()
df_nm=df_nm[pd.isnull(df_nm['id'])==0]
df_nm=df_nm[pd.isnull(df_nm['transid'])==0]
df_nm=df_nm[pd.isnull(df_nm['amount'])==0]
df_nm=df_nm[pd.isnull(df_nm['desc'])==0]
df_nm=df_nm[pd.isnull(df_nm['date'])==0]

df_1=df_nm.copy()

del df_1['id']
del df_1['transid']

df_1['date'].replace(r'(.*)T.*',r'\1',regex=True,inplace=True)
l1=[]
df_weekday={}
for i in range(len(df_1)):
    s=datetime.datetime.strptime(df_1.iloc[i]['date'], '%Y-%m-%d').strftime('%u')
    if s not in df_weekday:
        df_weekday[s]=1
    else:
        df_weekday[s]+=1
    l1.append(s)

l1=list(map(int, l1))
df_1['weekday']=l1

df_2_na=df_1.copy()


df_2_na['word1']=0
df_2_na['word2']=0
df_2_na['word3']=0
df_2_na['word4']=0
df_2_na['tf']=0
df_2_na['amount150']=0

df_2_na['word1'][df_2_na['desc'].str.lower().str.match(r'(.*dir.*dep.*)|(.*dep.*dir.*)')==True]=1
df_2_na['word2'][df_2_na['desc'].str.lower().str.match(r'.*payroll.*')==True]=1
df_2_na['word3'][df_2_na['desc'].str.lower().str.match(r'.*ach.*')==True]=1
df_2_na['word4'][df_2_na['desc'].str.lower().str.match(r'.*ppd.*')==True]=1
df_2_na['tf'][df_2_na['weekday'] ==4]=1
df_2_na['tf'][df_2_na['weekday'] ==5]=1
df_2_na['amount150'][df_2_na['amount']>=150]=1



features_na=df_2_na[['word1','word2','word3','word4','tf','amount150']]
features_na=features_na.as_matrix()


coef=np.array([0.20811551,  0.22613664,  0.10343929,  0.08940863,  0.19846145, 0.49991104])
intercept=np.array(-0.35308629)



def sigmoid(x):

    return 1 / (1 + math.e ** -x)

result=sigmoid(np.dot(features_na,(coef.T))+intercept)
result=result.tolist()

df_nm['result']=result
df_nm['result'][df_nm['result']>=0.5]=1
df_nm['result'][df_nm['result']<0.5]=0
df_nm['result'][df_nm['result']==1]='YES'
df_nm['result'][df_nm['result']==0]='NO'
df_nm.to_csv('Output.csv',index=False)
df_trans_missing.to_csv('Missing.csv',index=False)

print('Writing output to Output.csv file...\n\n')
print('Writing missing values to Missing.csv file...\n\n')
print('Success! Quitting...\n\n')

quit()
