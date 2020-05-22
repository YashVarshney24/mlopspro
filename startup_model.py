#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


dataset = pd.read_csv("50_Startups.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.columns


# In[6]:


y = dataset["Profit"]


# In[7]:


X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]


# In[8]:


X2=pd.get_dummies(X , drop_first=True)


# In[9]:


X2.head(2)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.30)


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


model=LinearRegression()


# In[14]:


model.fit(X_train,y_train)


# In[40]:


accuracy = model.score(X_test , y_test)


# In[41]:


print(accuracy)


# In[ ]:




