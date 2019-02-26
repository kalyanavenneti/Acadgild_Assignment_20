
# coding: utf-8

# # Machine Learning_Assignment 2

# In[17]:


#Here is the code to load the data 
import numpy as np 
import pandas as pd 
import scipy.stats as stats 
import matplotlib.pyplot as plt 
import sklearn 
from sklearn.datasets import load_boston 
from sklearn.linear_model import LinearRegression


# In[18]:


#loading the dataset into boston object
from sklearn.datasets import load_boston
boston = load_boston()


# In[19]:


#loading the data into bos dataframe
bos = pd.DataFrame(boston.data)
bos.head()


# In[20]:


#renaming the columns with it's feature names
bos.columns = boston.feature_names

#adding the price column to the bos
bos['PRICE'] = boston.target
bos.head()


# In[30]:


#differentiating independent features and storing them in X 
X = bos.iloc[:, :-1].values
print('X', X)


# In[31]:


#differentiating dependent(output) features from dataframe and storing them in Y
y = bos.iloc[:, -1].values
print('y', y)


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)


# In[33]:


print('X_train', X_train)


# In[32]:


print('X_test', X_test)


# In[34]:


print('y_train', y_train)


# In[35]:


print('y_test', y_test)


# In[24]:


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[25]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[36]:


print('y_pred', y_pred)


# In[28]:


#calculating the accuracy of model using r2_score 
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print('the predicted test score is:', score)


# In[26]:


#plotting the expected value v/s predicted value
plt.scatter(y_test,y_pred,color='blue')
plt.title('expected value v/s predicted value')
plt.xlabel('expected value')
plt.ylabel('predicted value')
plt.show()

