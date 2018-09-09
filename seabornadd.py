
# coding: utf-8

# In[2]:


import sklearn
from sklearn.datasets import load_iris
import pandas as pd


# In[9]:


import seaborn as sb
import matplotlib.pyplot as plt


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


iris=load_iris()


# In[4]:


data=pd.DataFrame(iris.data,columns=iris.feature_names)
label=pd.DataFrame(list(map(lambda x:iris.target_names[x],iris.target)),columns=['Spacies'])
iris=pd.concat([data,label],axis=1)
print(iris.head())


# ## 1 Use displot()
# 

# In[14]:


plt.subplot(2,2,1)
sb.distplot(iris['sepal length (cm)'],hist=True,kde=True,color='r')
plt.subplot(2,2,2)
sb.distplot(iris['sepal width (cm)'],hist=True,kde=False,color='b')
plt.subplot(2,2,3)
sb.distplot(iris['petal length (cm)'],hist=True,kde=False,color='g')
plt.subplot(2,2,4)
sb.distplot(iris['petal width (cm)'],hist=True,kde=True,color='y')


# ## 2 boxplot

# In[15]:


sb.boxplot(data=iris,palette='rainbow')


# ## 3 countplot

# In[26]:


sb.countplot(x='Spacies',data=iris)


# ## 4 pairplot

# In[27]:


sb.pairplot(iris,palette='coolwarm')


# ## 5 lmplot

# In[28]:


sb.lmplot('sepal length (cm)','petal length (cm)' ,data=iris,hue='Spacies')


# ## 6 barplot

# In[30]:


sb.barplot('Spacies','sepal length (cm)',data=iris)


# ## 7 heatmap

# In[38]:



sb.heatmap(iris.corr())

