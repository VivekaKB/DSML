#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine



# In[8]:


a=load_wine()
x=a.data
y=a.target


# In[9]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)  


# In[10]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
prediction=gnb.predict(x_test)
print(prediction)


# In[16]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, prediction)*100)


# In[11]:


from sklearn import metrics
print(metrics.accuracy_score(y_test,prediction))


# In[17]:


from sklearn.metrics import confusion_matrix
cm=np.array(confusion_matrix(y_test,prediction))
cm=confusion_matrix(y_test,prediction)
print(cm)


# In[18]:


plt.plot(x,y)
plt.show()


# In[ ]:




