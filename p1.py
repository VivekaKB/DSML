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


# In[ ]:




