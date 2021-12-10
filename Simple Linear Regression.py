#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[31]:


dataset = pd.read_csv('salary.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
x


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 42)


# In[ ]:





# In[34]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[36]:


y_pred = regressor.predict(x_test)


# In[43]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience {Test set}')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




# In[ ]:




