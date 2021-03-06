#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('iris.csv')
print(df.head())


# In[2]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['variety'].value_counts()


# In[9]:


df.isnull().sum()


# In[10]:


df['sepal.length'].hist()


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris=pd.read_csv('iris.csv')
iris.plot(kind ="scatter",
          x ='sepal.length',
          y ='petal.length')
plt.grid()


# In[12]:


iris.plot(kind='scatter', x='petal.length', y='petal.width') 
plt.show()


# In[13]:


a= iris['variety'].value_counts()
species = a.index
count = a.values
plt.bar(species,count,color = 'lightgreen')
plt.xlabel('species')
plt.ylabel('count')
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split , KFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from collections import Counter


# In[15]:


iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])
iris_df.head()


# In[16]:


iris_df.describe()


# In[17]:


x= iris_df.iloc[:, :-1]
y= iris_df.iloc[:, -1]


# In[18]:


x.head()


# In[19]:


y.head()


# In[20]:


x_train, x_test, y_train, y_test= train_test_split(x, y,
                                                   test_size= 0.2,
                                                   shuffle= True, 
                                                   random_state= 0)
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)

x_test= np.asarray(x_test)
y_test= np.asarray(y_test)

                                            


# In[21]:


print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')


# In[22]:


scaler= Normalizer().fit(x_train) 
normalized_x_train= scaler.transform(x_train) 
normalized_x_test= scaler.transform(x_test)


# In[23]:


print('x train before Normalization')
print(x_train[0:5])
print('\nx train after Normalization')
print(normalized_x_train[0:5])


# In[24]:


di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0:'Virginica'}
before= sns.pairplot(iris_df.replace({'target': di}), hue= 'target')
before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)

## After
iris_df_2= pd.DataFrame(data= np.c_[normalized_x_train, y_train],
                        columns= iris['feature_names'] + ['target'])
di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
after= sns.pairplot(iris_df_2.replace({'target':di}), hue= 'target')
after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)


# In[25]:


def distance_ecu(x_train, x_test_point):
  
  distances= []
  for row in range(len(x_train)):
      current_train_point= x_train[row] 
      current_distance= 0 

      for col in range(len(current_train_point)): 
          
          current_distance += (current_train_point[col] - x_test_point[col]) **2
        
      current_distance= np.sqrt(current_distance)

      distances.append(current_distance)


  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances


# In[26]:


def nearest_neighbors(distance_point, K):
   

   
    df_nearest= distance_point.sort_values(by=['dist'], axis=0)

   
    df_nearest= df_nearest[:K]
    return df_nearest


# In[27]:


def voting(df_nearest, y_train):
   
    
    counter_vote= Counter(y_train[df_nearest.index])

    y_pred= counter_vote.most_common()[0][0]  

    return y_pred


# In[28]:


def KNN_from_scratch(x_train, y_train, x_test, K):

   

    y_pred=[]

   
    for x_test_point in x_test:
      distance_point  = distance_ecu(x_train, x_test_point) 
      df_nearest_point= nearest_neighbors(distance_point, K) 
      y_pred_point    = voting(df_nearest_point, y_train) 
      y_pred.append(y_pred_point)

    return y_pred  


# In[29]:


K=3
y_pred_scratch= KNN_from_scratch(normalized_x_train, y_train, normalized_x_test, K)
print(y_pred_scratch)


# In[30]:


knn=KNeighborsClassifier(K)
knn.fit(normalized_x_train, y_train)
y_pred_sklearn= knn.predict(normalized_x_test)
print(y_pred_sklearn)


# In[31]:


print(np.array_equal(y_pred_sklearn, y_pred_scratch))


# In[32]:


print(f'The accuracy of our implementation is {accuracy_score(y_test, y_pred_scratch)}')
print(f'The accuracy of sklearn implementation is {accuracy_score(y_test, y_pred_sklearn)}')


# In[33]:


n_splits= 4 
kf= KFold(n_splits= n_splits) 

accuracy_k= [] 
k_values= list(range(1,30,2)) 

for k in k_values:
  accuracy_fold= 0
  for normalized_x_train_fold_idx, normalized_x_valid_fold_idx in  kf.split(normalized_x_train):
      normalized_x_train_fold= normalized_x_train[normalized_x_train_fold_idx] 
      y_train_fold= y_train[normalized_x_train_fold_idx]

      normalized_x_test_fold= normalized_x_train[normalized_x_valid_fold_idx]
      y_valid_fold= y_train[normalized_x_valid_fold_idx]
      y_pred_fold= KNN_from_scratch(normalized_x_train_fold, y_train_fold, normalized_x_test_fold, k)

      accuracy_fold+= accuracy_score (y_pred_fold, y_valid_fold) 
  accuracy_fold= accuracy_fold/ n_splits 
  accuracy_k.append(accuracy_fold)


# In[34]:


print(f'The accuracy for each K value was {list ( zip (accuracy_k, k_values))}') 


# In[35]:


print(f'Best accuracy was {np.max(accuracy_k)}, which corresponds to a value of K= {k_values[np.argmax(accuracy_k)]}')


# In[ ]:




