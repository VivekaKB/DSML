#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


# In[4]:


a= load_iris()
x=a.data
y=a.target


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[18]:


y_pred = clf.predict(x_test)
print(y_pred)


# In[17]:


print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))


# In[14]:


from sklearn import tree
tree.plot_tree(clf)


# In[ ]:




