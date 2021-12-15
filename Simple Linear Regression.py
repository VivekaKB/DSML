#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset = pd.read_csv('salary.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
x


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 42)


# In[ ]:





# In[5]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[6]:


y_pred = regressor.predict(x_test)
y_pred


# In[9]:



z=regressor.predict([[12]])
print(z)


# In[43]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience {Test set}')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[10]:


from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
print(f'R2 score: {score}')


# In[ ]:





# In[ ]:




