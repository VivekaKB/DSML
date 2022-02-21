#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pd
from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()
x=cancer.data
y=cancer.target


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=101)


# In[5]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)


# In[6]:


predictions = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[7]:


print(classification_report(y_test,predictions))


# In[ ]:




