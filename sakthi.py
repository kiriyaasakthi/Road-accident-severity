import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv("/kaggle/input/road-accidents-severity-dataset/RTA Dataset.csv")
data.head()
skew=data.skew()
print(skew)
correlation = data.corr()
sns.heatmap(correlation,annot = True, cmap = 'Blues')
px.pie(data, data['Sex_of_driver'], data['Number_of_casualties'], template='ggplot2',hole=0.5)
px.violin(data,data['Road_surface_type'], data['Number_of_casualties'],color='Accident_severity',template='ggplot2')
px.violin(data,data['Road_surface_type'], data['Number_of_casualties'],color='Accident_severity',template='ggplot2')
px.pie(data,data['Cause_of_accident'],data['Number_of_casualties'],color='Cause_of_accident',template='ggplot2',hole=0.35)
px.histogram(data,data['Day_of_week'],data['Number_of_casualties'],color='Day_of_week',template='ggplot2')
px.histogram(data,data['Educational_level'],data['Number_of_casualties'],color='Educational_level',template='ggplot2')
px.histogram(data,data['Vehicle_driver_relation'],data['Number_of_casualties'],color='Vehicle_driver_relation',template='ggplot2')
df.drop(['Time','Driving_experience','Type_of_vehicle','Educational_level'],axis=1,inplace=True)
df.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = l.fit_transform(df[col])
x=df.drop('Accident_severity',axis=1)
y=df['Accident_severity']
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30)
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
xtrain=mms.fit_transform(xtrain)
xtest=mms.fit_transform(xtest)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
rf.fit(xtrain,ytrain)
    
ypred=rf.predict(xtest)

print('-----------------------------------------------------------------------------------------------------------------------')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score

print('confusion matrix :',confusion_matrix(ytest,ypred))
print('classification report:',classification_report(ytest,ypred))
print('accuracy :',round(accuracy_score(ytest,ypred),2))
print('precision :',round(precision_score(ytest,ypred,average='weighted'),2))
print('recall :',round(recall_score(ytest,ypred,average='weighted'),2))
print('f1 :',round(f1_score(ytest,ypred,average='weighted'),2))
print()

