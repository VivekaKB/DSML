#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score


# In[3]:


a= load_digits()
x=a.data
y=a.target


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)


# In[5]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[6]:


y_pred = clf.predict(x_test)
print(y_pred)


# In[7]:


print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))


# In[8]:


from sklearn import tree
tree.plot_tree(clf)


# In[ ]:




