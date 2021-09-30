#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#for interactivity
from ipywidgets import interact


# In[6]:


df= pd.read_csv('data.xlsx.csv')


# In[8]:


df.shape


# In[7]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df['label'].value_counts()


# In[11]:


df.describe()


# In[13]:


@interact
def summary(crops = list(df['label'].value_counts().index)):
    x = df[df['label']== crops]
    z = df.drop(['label'],axis=1)
    y = []
    y.append(z)
    for i in z:
        print('Minimum', i, 'required', x[i].min())
        print('Average', i, 'required', x[i].mean())
        print('Maximum', i, 'required', x[i].max())
        print('--------------------------------------------------')


# In[14]:


print('crops which requires very high ratio of Nitrogen content in soil:', df[df['N']>120]['label'].unique())
print('crops which requires very high ratio of Phosphorous content in soil:', df[df['P']>100]['label'].unique())
print('crops which requires very high ratio of Potassium content in soil:', df[df['K']>200]['label'].unique())
print('crops which requires very high rainfall:', df[df['rainfall']>200]['label'].unique())
print('crops which requires very low temperature:', df[df['temperature']<10]['label'].unique())
print('crops which requires very high temperature:', df[df['temperature']>40]['label'].unique())
print('crops which requires very low humidity:', df[df['humidity']<20]['label'].unique())
print('crops which requires very low ph:', df[df['ph']<4]['label'].unique())
print('crops which requires very high ph:', df[df['ph']>9]['label'].unique())


# In[15]:


print('Summer Crops:', df[(df['temperature']>30) & (df['humidity']>50)]['label'].unique())
print('Winter Crops:', df[(df['temperature']<20) & (df['humidity']>30)]['label'].unique())
print('Rainy Crops:', df[(df['rainfall']>200) & (df['humidity']>30)]['label'].unique())


# In[16]:


z = df.drop(['label'],axis=1)
z = df.loc[:,z.columns].values
x_df = pd.DataFrame(z)
x_df.head()


# In[17]:


#Determine Optimum number of cluster by elbow method
from sklearn.cluster import KMeans
plt.rcParams['figure.figsize'] = (10,4)
wcss = []
for i in range (1,11):
    km = KMeans(n_clusters =i, init= 'k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(z)
    wcss.append(km.inertia_)
    
#plot the results
plt.plot(range(1,11), wcss)
plt.title('Elbow Method', fontsize= 15)
plt.xlabel('No. of cluster')
plt.ylabel('wcss')
plt.show()


# In[18]:


plt.figure(figsize=(15, 28))
plt.subplot(4,2,1)
sns.barplot(df['N'], df['label'])
plt.ylabel(' ')
plt.xlabel('Nitrogen')
plt.subplot(4,2,2)
sns.barplot(df['P'], df['label'])
plt.ylabel(' ')
plt.xlabel('Phophorus')
plt.subplot(4,2,3)
sns.barplot(df['K'], df['label'])
plt.ylabel(' ')
plt.xlabel('Potassium')
plt.subplot(4,2,4)
sns.barplot(df['temperature'], df['label'])
plt.ylabel(' ')
plt.xlabel('Temperature')
plt.subplot(4,2,5)
sns.barplot(df['humidity'], df['label'])
plt.ylabel(' ')
plt.xlabel('Humidity')
plt.subplot(4,2,6)
sns.barplot(df['ph'], df['label'])
plt.ylabel(' ')
plt.xlabel('pH')
plt.subplot(4,2,7)
sns.barplot(df['rainfall'], df['label'])
plt.ylabel(' ')
plt.xlabel('rainfall')
#apply for loop


# # Observation
# 
# 1. Cotton requires high amount of Nitrogen among all
# 2. Grapes and Apple requires very high amount of phosphorus and Potassium
# 3. least amount of potassium is the favorable condition of Orange to grow
# 4. Papaya requires more than 30 degree to grow well whereas others required <= 30 degree
# 5. chickpea and kidneybeans humidity requires very less humidity to grow
# 6. All crops require more than pH value of 5 to grow
# 7. Rice requires very heavy rainfall (more than 200mm) where the muskmelon requires the least

# In[19]:


x = df.drop(['label'], axis=1)
x.head()


# In[20]:


y = df['label']
y.head()


# In[21]:


print('shape of x:', x.shape)
print('shape of y:', y.shape)


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[23]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(x_train,y_train)
y_pred1 = clf_knn.predict(x_test)
print("Accuracy Score of KNN:",accuracy_score(y_test,y_pred1))
print('--------------------------------------------------')

from sklearn.svm import SVC
clf_svc = SVC()
clf_svc.fit(x_train,y_train)
y_pred2 = clf_svc.predict(x_test)
print("Accuracy Score of SVC:",accuracy_score(y_test,y_pred2))
print('--------------------------------------------------')

from sklearn.tree import DecisionTreeClassifier
clf_dtc = DecisionTreeClassifier(criterion='entropy',random_state=7)
clf_dtc.fit(x_train,y_train)
y_pred3 = clf_dtc.predict(x_test)
print("Accuracy Score of decision tree:",accuracy_score(y_test,y_pred3))
print('--------------------------------------------------')

from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(random_state=1)
clf_rfc.fit(x_train, y_train)
y_pred4 = clf_rfc.predict(x_test)
print("Accuracy Score of Random Forest:",accuracy_score(y_test,y_pred4))
print('--------------------------------------------------')


# In[26]:


y_train_pred = clf_rfc.predict(x_train)
print("Accuracy Score of Random Forest:",accuracy_score(y_train,y_train_pred))


# In[28]:


output = pd.DataFrame({'Real_class': y_test, 'Predicted_class': y_pred4})
output.head()


# In[30]:


#TO CHECK THE MODEL PUT VALUES AND SEE WHICH CROP GROWS WITH GIVEN VALUES

input = np.array([[90,
                   42,
                   43,
                   20,
                   82,
                   6,
                   200]])
clf_rfc.predict(input)


# In[31]:


#TO CHECK THE MODEL PUT VALUES AND SEE WHICH CROP GROWS WITH GIVEN VALUES

input = np.array([[50,
                   42,
                   43,
                   40,
                   40,
                   4,
                   50]])
clf_rfc.predict(input)


# In[32]:


df.head()


# In[34]:


input=np.array([[65,
                52,
                35,
                35,
                 75,
                 9,
                150]])

clf_rfc.predict(input)


# In[ ]:




