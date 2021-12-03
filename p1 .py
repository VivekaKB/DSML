#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
a=load_digits()
x=a.data
y=a.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
KNeighborsClassifier(n_neighbors=6)
p=(knn.predict(x_test))
print(p)


# In[5]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,p))
print(classification_report(y_test,p))


# In[13]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
 
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)
 


 
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()



# In[ ]:





# In[ ]:




