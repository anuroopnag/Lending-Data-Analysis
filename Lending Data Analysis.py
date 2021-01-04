#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('loan_data.csv')
df.head()


# In[3]:


df['not.fully.paid'].value_counts()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


sns.pairplot(df,hue = 'not.fully.paid')


# In[8]:


df.columns


# In[11]:


df['not.fully.paid'].count()


# In[10]:


df.columns


# In[14]:


df['not.fully.paid'].hist()


# In[21]:


plt.figure(figsize=(10,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=df)


# In[22]:


plt.figure(figsize=(10,7))
sns.countplot(x='purpose',hue='credit.policy',data=df)


# In[23]:


sns.scatterplot(x='int.rate',y='fico',data=df)


# In[26]:


sns.lmplot(x='fico',y='int.rate',data=df,hue='credit.policy',col='not.fully.paid')


# In[27]:


df.head()


# In[32]:


new_purpose=['purpose']
loans=pd.get_dummies(data=df,columns=new_purpose,drop_first=True)
loans.head()


# In[38]:


purp = pd.get_dummies(df['purpose'],drop_first=True)
purp


# In[40]:


from sklearn.model_selection import train_test_split
X = loans.drop('not.fully.paid',axis=1)
y = loans['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[45]:


predictions = dt.predict(X_test)
predictions


# In[47]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))


# In[57]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train,y_train)
predict = rf.predict(X_test)


# In[58]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
print('\n')
print(confusion_matrix(y_test,predict))


# In this case even after creating an expansive Random Forest Algorithm/model a susbstantial difference couldnot be seen probabaly because the data frm it's inception was not balanced as it should have been.The data propreity here is in question as without a balanced data the conclusion would be tough to estimate.
# Although the model perfomed well even with this unbalanced data and showed a precision of 76% but a better dataset would have been much more beneficial
