#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #For mathematic evaluation
import pandas as pd #For data handling
import matplotlib as mlp #For data visualisation
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 
from pylab import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv(r'C:\Users\shashank\Desktop\data\Employee Churn/TelcoCustomerChurn.csv')


# In[18]:


data.head(5)


# In[16]:


# Data to plot
sizes = data['Churn'].value_counts(sort = True)
colors = ["grey","purple"] 
rcParams['figure.figsize'] = 5,5
labels = ["No","Yes"]
explode = (0.1,0.2) 
# Plot
plt.pie(sizes, explode=explode, labels=labels ,colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Churn in Dataset')
plt.show()
print("Above pie chart shows around 27% of employees churning out")


# In[4]:


data.drop(['customerID'], axis=1, inplace=True)


# In[13]:


data.loc[data.Churn=="No","Churn"] = 0
data.loc[data.Churn=="Yes",'Churn'] = 1
data.head(4)


# In[6]:


data.loc[data.PhoneService=="Yes","PhoneService"] = 1
data.loc[data.PhoneService=="No","PhoneService"] = 0
data.head(4)


# In[ ]:


data.groupby('gender').Churn.mean()


# In[ ]:


data.groupby("Dependents").Churn.mean()


# In[ ]:



data.groupby("OnlineSecurity").Churn.mean()


# In[ ]:



data.groupby("InternetService").Churn.mean()


# In[ ]:



data.groupby("MultipleLines").Churn.mean()


# In[ ]:



data.groupby("PhoneService").Churn.mean()


# In[ ]:



data.groupby("tenure").Churn.mean()


# In[ ]:


data.groupby("Partner").Churn.mean()


# In[7]:


data.drop(["PhoneService"], axis=1, inplace=True)


# In[8]:


data.drop(["MultipleLines"], axis=1 , inplace=True)


# In[9]:


data['MonthlyCharges'] = pd.to_numeric(data['MonthlyCharges'])


# In[14]:


data.drop(["TotalCharges"], axis=1, inplace=True)


# In[12]:


data.drop(['PaymentMethod'], axis=1, inplace=True)


# In[15]:


data.info()


# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dummy_columns = ['gender']

for column in data.columns:
     if data[column].dtype == object and column != 'customerID':
        if data[column].nunique() == 2:
            #apply Label Encoder for binary ones
            data[column] = le.fit_transform(data[column]) 
        else:
            dummy_columns.append(column)

data = pd.get_dummies(data = data, columns = dummy_columns)            


# In[39]:


data.head()


# In[18]:


Y = data["Churn"].values
X = data.drop(labels = ["Churn"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=101)


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train,Y_train)


# In[17]:


from sklearn import metrics
prediction_test = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, prediction_test)*100
accuracy = round(accuracy)
print("The accuracy for our given model is: {} %",accuracy)

weights = pd.Series(model.coef_[0], index=X.columns.values)
weights.sort_values(ascending=False)