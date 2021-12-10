#!/usr/bin/env python
# coding: utf-8

# In[27]:


from sklearn.datasets import load_diabetes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


d = load_diabetes()
X = d.data
y = d.target


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[30]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
prediction = gnb.predict(X_test)
print(prediction)


# In[32]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, prediction)*100)


# In[33]:


cm=np.array(confusion_matrix(y_test,prediction))
print(cm)


# In[36]:


plt.plot(X,y)
plt.show()


# In[ ]:




