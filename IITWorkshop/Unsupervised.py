
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[17]:


df = pd.DataFrame({'x':[12,20,28,18,29,33,24,45,52,51,53,55,57,62,64,68,77,71],
                  'y' : [39,39,30,52,54,46,55,59,63,70,66,62,58,23,14,8,19,7]})


# In[23]:


centroids = {i+1 : [np.random.randint(0,80),np.random.randint(0,80)]
            for i in range(3)}


# In[24]:


centroids


# In[25]:


plt.scatter(df['x'],df['y'])
colmap = {1:'r', 2:'g', 3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.show()


# In[26]:


#Assignment Stage
def assignment(df,centroids):
    for i in centroids.keys():
        df['distance_{}'.format(i)] = np.sqrt((df['x']-centroids[i][0])**2 + (df['y']-centroids[i][1])**2)
    df['closest']=df.loc[:,'distance_1':'distance_3'].idxmin(axis=1)
    df['closest']=df['closest'].map(lambda x: int(x.lstrip('distance_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df
df = assignment(df,centroids)
df.head()


# In[27]:


plt.scatter(df['x'],df['y'],color=df['color'])
plt.show()


# In[35]:


#Update Stage
import copy
old_centeroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0]=np.mean(df[df['closest']==i]['x'])
        centroids[i][1]=np.mean(df[df['closest']==i]['y'])
    return k
centroids =update(centroids)


# In[36]:


centroids


# In[37]:


df = assignment(df,centroids)


# In[38]:


plt.scatter(df['x'],df['y'],color=df['color'])
plt.show()


# In[31]:


df[df['closest']==1]


# In[39]:


ds = pd.DataFrame({'x':[12,20,28,18,29,33,24,45,52,51,53,55,57,62,64,68,77,71],
                  'y' : [39,39,30,52,54,46,55,59,63,70,66,62,58,23,14,8,19,7]})


# In[40]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3, max_iter=500)
kmeans.fit(ds)


# In[41]:


labels=kmeans.predict(ds)
centroids=kmeans.cluster_centers_


# In[44]:


centroids


# In[46]:


cost =[] 
for i in range(1, 11): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(ds)     
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_) 


# In[47]:


# plot the cost against K values 
plt.plot(np.arange(1, 11), np.array(cost)) 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
plt.show() # clear the plot 


# In[ ]:




