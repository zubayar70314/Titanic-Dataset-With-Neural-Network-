#!/usr/bin/env python
# coding: utf-8

# #  Titanic Dataset  with Neural Network
# 
# Serajus Salehin, ID : 2017-3-60-018
# Zubayar Mahatab Md Sakif, ID :2018-1-60-105
# Israt Jahan Mridula, ID : 2017-1-60-108
# Md. Nazmul islam,ID : 2017-2-60-38 
# 
# 

# In[529]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Neural Network
#from tensorflow import keras
import keras

from keras.models import Sequential 
from keras.layers import Dense

# load the data
df_train = pd.read_csv('/content/train.csv')
df_test = pd.read_csv('/content/test.csv')
df = df_train.append(df_test , ignore_index = True)

print(df_train.shape, df_test.shape,df.shape)


# ### Pclass Analysis

# In[530]:


df['Pclass'].isnull().sum()


# In[531]:


df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# ### Name To Title

# In[532]:


df.Name.head(10)


# In[533]:


df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

df['Title'].value_counts()


# In[534]:


df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 
             & (df.Title !=  'Mrs')] = 'Others'

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[535]:


df['Title'].value_counts()


# In[536]:


df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)


# ### Sex(Male:0 Female:1)

# In[537]:


df.Sex.isnull().sum()


# In[538]:


df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[539]:


df.Sex = df.Sex.map({'male':0, 'female':1})


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ### Cabin(Dropped)

# In[540]:


df.Cabin.isnull().sum(axis=0)


# In[541]:


df = df.drop(labels=['Cabin'], axis=1)


# ### Embarked(Dropped)

# In[542]:


df.Embarked.isnull().sum(axis=0)


# In[543]:


df['Embarked'].value_counts()


# In[544]:


df.Embarked.fillna('S' , inplace=True )


# In[545]:


df[['Embarked', 'Survived','Pclass', 'Age']].groupby(['Embarked'], as_index=False).mean()


# In[546]:


df = df.drop(labels='Embarked', axis=1)


# ### Age(RandomForestRegressor)

# In[547]:


df.Age.isnull().sum()


# In[547]:





# In[548]:


df[['Title', 'Age']].groupby(['Title']).mean()


# In[549]:


df[['Title', 'Age']].groupby(['Title']).std()


# In[550]:


df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','SibSp']]

X_train  = df_sub.dropna().drop('Age', axis=1)
y_train  = df['Age'].dropna()
X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train, y_train)
y_pred = np.round(regressor.predict(X_test),1)
df.Age.loc[df.Age.isnull()] = y_pred
df.Age.isnull().sum(axis=0) 


# ### Ticket Analysis

# In[551]:


df.Ticket.isnull().sum()


# In[552]:


df.Ticket.head(20)


# In[553]:


df.Ticket = df.Ticket.map(lambda x: x[0])
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[554]:


df['Ticket'].value_counts()


# In[555]:


df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], 'others')

df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[556]:


df = pd.get_dummies(df,columns=['Ticket'])


# ## Trainning and Testing
# 

# In[557]:


#from sklearn.model_selection import train_test_split
#df = df.drop(labels=['SibSp','Parch','Fare','Title','PassengerId'], axis=1)
#y_true=df["Survived"]
#X_train,X_test,y_train,y_test = train_test_split(df,y_true,test_size=0.2,random_state=0)


# In[558]:


df = df.drop(labels=['SibSp','Parch','Fare','Title','PassengerId'], axis=1)
y_train = df[0:891]['Survived'].values
X_train = df[0:891].drop(['Survived'], axis=1).values
y_test  = df[891:].drop(['Survived'], axis=1).values


# In[559]:


#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

#def Model(model,name):
#    model.fit(X_train,y_train)
#    score = model.score(X_test, y_test)
#    model_train_score = model.score(X_train, y_train)
#    model_test_score = model.score(X_test, y_test)
#    prediction = model.predict(X_test)
#    print('{} Trainng Score {}\n'.format(name,model_train_score))
#    print('{} Testing Score {}\n'.format(name,model_test_score))
#    print('{} Testing Score {}\n'.format(name,score))


# In[560]:


#from sklearn import svm
#clf = svm.SVC()
#Model(clf,"SVM")


# In[561]:



#from sklearn.linear_model import LogisticRegression
#lr=LogisticRegression()
#Model(lr,"Logistic Regression")


# In[562]:


from keras.initializers import glorot_uniform
model = Sequential()
model.add(Dense(9, activation = 'relu', input_dim = 16))
model.add(Dense(9,  activation = 'relu'))
model.add(Dense(5,  activation = 'relu'))
model.add(Dense(1,  activation = 'sigmoid'))


# In[563]:


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 32 , epochs = 500)


# ##Prediction

# In[564]:


#y_pred = model.predict(X_test,batch_size=32)
#y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

#predicted_table = pd.DataFrame({'PassengerId': df_test['PassengerId'],'Age': df_test['Age'],'Sex': df_test['Sex'], 'Survived': y_final})
#predicted_table.to_csv('/content/predicted.csv', index=False)

