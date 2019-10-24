#!/usr/bin/env python
# coding: utf-8

# In[280]:


# Is project


# In[281]:


#imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


# # Feature Engineering

# In[282]:


data = pd.read_csv('data.csv')


# In[283]:


data.head()


# In[284]:


data.dtypes


# In[285]:


data.isnull().sum()


# ### Null values

# #### Grade column

# In[286]:


data['Grade'].fillna('B' ,inplace=True)


# In[287]:


data['Grade'].value_counts()


# #### School category

# In[288]:


data['SchoolCategory'].value_counts()


# In[289]:


data['SchoolCategory'].fillna('private', inplace=True)


# #### Mother Alive

# In[290]:


data['MotherAlive'] = np.where((data['MothersEmployment'].notna()), #1 means alive
                                   1,data['MotherAlive'])


# In[291]:


data['MotherAlive'] = np.where(data['MothersEmployment']=="deceased", #0 means deceased
                                   0,data['MotherAlive'])


# In[292]:


data['MotherAlive'].fillna(2, inplace=True) #2 means it was left blank


# #### Father Alive

# In[293]:


data['FatherAlive'] = np.where((data['FatherEmployment'].notna()),  #1 means alive
                                   1,data['FatherAlive'])


# In[294]:


data['FatherAlive'] = np.where(data['FatherEmployment']=="deceased",  #0 means deceased
                                   0,data['FatherAlive'])


# In[295]:


data['FatherAlive'].fillna(2, inplace=True) #2 means it was left blank


# #### fee balance

# In[296]:


data['FeeBalance'].fillna(0, inplace=True)


# #### siblings

# In[297]:


data['Siblings'] = np.where((data['SiblingsFees'].isnull()),
                                   0,data['Siblings'])


# In[298]:


smode_filler = data['Siblings'].mode()


# In[299]:


data['Siblings'].fillna(smode_filler, inplace=True)


# In[300]:


data['Siblings'].fillna(0, inplace= True)


# #### siblings fees

# In[301]:


data['SiblingsFees'] = np.where(data['Siblings']==0,
                                   0,data['SiblingsFees'])


# #### Payment  Mode

# In[302]:


data['PaymentMode'].value_counts()


# In[303]:


data['PaymentMode'].fillna('Parent', inplace=True)


# #### Father Employment Status

# In[304]:


data['FatherEmployment'].value_counts()


# In[305]:


data['FatherEmployment'] = np.where(data['FatherAlive']==0,  #0 means deceased
                                   "blank",data['FatherEmployment'])


# In[306]:


data['FatherEmployment'] = np.where(data['FatherAlive']==2,  #2 means fatherAlive was left blank
                                   "blank",data['FatherEmployment'])


# In[307]:


data['FatherEmployment'].fillna('employed', inplace = True)


# #### Mother Employment Status

# In[308]:


data['MothersEmployment'].value_counts()


# In[309]:


data['MothersEmployment'] = np.where(data['MotherAlive']==0,  #0 means deceased
                                   "blank",data['MothersEmployment'])


# In[310]:


data['MothersEmployment'] = np.where(data['MotherAlive']==2,  #2 means MotherAlive was left blank
                                   "blank",data['MothersEmployment'])


# In[311]:


data['MothersEmployment'].fillna ('employed',inplace = True)


# #### Guardian

# In[312]:


data['Guardian'].value_counts()


# In[313]:


data['Guardian'].fillna('none', inplace = True)


# #### Guardian Employment

# In[314]:


data['GuardianEmployment'].value_counts()


# In[315]:


data['GuardianEmployment'] = np.where(data['Guardian']=="none",  
                                   "no",data['GuardianEmployment'])


# In[316]:


data['GuardianEmployment'].fillna('no', inplace= True)


# #### Financial Income

# In[317]:


data['MotherFinancialIncome'] = np.where(data['MothersEmployment']=="deceased",
                                   0,data['MotherFinancialIncome'])


# In[318]:


data['FatherFinancialIncome'] = np.where(data['FatherEmployment']=="deceased",
                                   0,data['FatherFinancialIncome'])


# In[319]:


data['FatherFinancialIncome'] = np.where(data['FatherEmployment']=="unemployed",
                                   0,data['FatherFinancialIncome'])


# In[320]:


median_filler = data['FatherFinancialIncome'].median()


# In[321]:


data['FatherFinancialIncome'].fillna(median_filler, inplace=True)


# #### Amount Able to pay

# In[322]:


amount_filler = data['Amount'].median()


# In[323]:


data['Amount'].fillna(amount_filler, inplace=True)


# In[324]:


data.isnull().sum()


# ### Encoding

# #### Courses

# In[325]:



le = LabelEncoder()
le.fit(data['Courses'])


# In[326]:


list(le.classes_)


# In[327]:


data.Courses = le.transform(data['Courses']) 


# In[328]:


data.head()


# #### Grade

# In[329]:


le.fit(data['Grade'])


# In[330]:


list(le.classes_)


# In[331]:


data.Grade = le.transform(data['Grade'])


# In[332]:


data.head()


# #### School Category

# In[333]:


le.fit(data['SchoolCategory'])


# In[334]:


list(le.classes_)


# In[335]:


data.SchoolCategory = le.transform(data['SchoolCategory'])


# In[336]:


data.head()


# #### Payment Mode

# In[337]:


le.fit(data['PaymentMode'])


# In[338]:


list(le.classes_)


# In[339]:


data.PaymentMode = le.transform(data['PaymentMode'])


# In[340]:


data.head()


# #### Father Employment

# In[341]:


le.fit(data['FatherEmployment'])


# In[342]:


list(le.classes_)


# In[343]:


data.FatherEmployment = le.transform(data['FatherEmployment'])


# #### Mother Employment

# In[344]:


le.fit(data['MothersEmployment'])


# In[345]:


list(le.classes_)


# In[346]:


data.MothersEmployment = le.transform(data['MothersEmployment'])


# #### Guardian Employment

# In[347]:


le.fit(data['GuardianEmployment'])


# In[348]:


list(le.classes_)


# In[349]:


data.GuardianEmployment = le.transform(data['GuardianEmployment'])


# #### Awarded

# In[350]:


le.fit(data.Awarded)


# In[351]:


list(le.classes_)


# In[352]:


data.Awarded = le.transform(data['Awarded'])


# #### Guardian

# In[353]:


le.fit(data.Guardian)


# In[354]:


list(le.classes_)


# In[355]:


data.Guardian = le.transform(data['Guardian'])


# In[356]:


data.head()


# # Model

# In[357]:


sns.heatmap(data.corr())


# In[358]:


#X = data.drop(columns=['Awarded'])
y=data['Awarded']


# In[359]:


#X=data[['Courses','Grade','FeesPer Year','FeeBalance','FatherEmployment','MothersEmployment','GuardianEmployment','FatherFinancialIncome','MotherFinancialIncome','Amount','SiblingsFees']]
X = data.drop(columns=['Awarded','PaymentMode','Year'])


# In[360]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[361]:


X.head()


# In[362]:


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=16))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


# In[363]:


#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


# In[373]:


#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=150)


# In[376]:


eval_model=classifier.evaluate(X_train, y_train)
eval_model


# In[377]:


y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)


# In[367]:


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)


# In[368]:


#((cm[0,0]+cm[1,1])/np.sum(cm))*100


# In[369]:


X.head()


# In[370]:


from numpy import array

#Xnew = array([[0,4,4,250000,0,4,3,0,1,1,3,3,100000,25000,2,29000,]])


# In[371]:


#ynew = classifier.predict_classes(Xnew)


# In[372]:


#print(ynew)


# In[379]:


from sklearn.externals import joblib

joblib.dump(classifier, 'model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")





