
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# use seaborn plotting style defaults
import seaborn as sns


# In[2]:


sns.set()
df = pd.read_csv("airquality.csv")
print('std is ', df.std())
#print('mean is ', df.mean())
#df = (df - df.mean())/df.std()
df.columns
#df.info()


# In[3]:


#bservations and variables
observations = list(df.index)
print(observations)
variables = list(df.columns)
print(variables)


# In[4]:


sns.boxplot(data=df, orient="v", palette="Set2")
#Covariance

#dfc = df - df.mean() #centered data
plt. figure()


# In[5]:


ax = sns.heatmap(df.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True, cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
#plt.title('Covariance matrix')


# In[6]:


#Principal component analysis

pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)
plt. figure()
plt.scatter(Z[:,0], Z[:,1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
for label, x, y in zip(observations,Z[:, 0],Z[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),textcoords='offset points', ha='right', va='bottom')


# In[7]:


#Eigenvectors
A = pca.components_ 
plt. figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$');
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),textcoords='offset points', ha='right', va='bottom') 

plt. figure()
plt.scatter(A[:, 0],A[:, 1],marker='o',c=A[:, 2],s=A[:, 3]*500, cmap=plt.get_cmap('Spectral'))

for label, x, y in zip(variables,A[:, 0],A[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),textcoords='offset points', ha='right', va='bottom',bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))



# In[8]:


#Eigenvalues
Lambda = pca.explained_variance_ 
#Scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda, 'ro-', lw=2)
plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
#plt.xlabel('Number of components')
plt.ylabel('Explained variance')


# In[9]:


#Explained variance
ell = pca.explained_variance_ratio_
plt. figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[10]:


#Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[0] 
A2 = A[1]
Z1 = Z[:,0] 
Z2 = Z[:,1]
plt. figure()
for i in range(len(A1)):
# arrows project features as vectors onto PC axes
    plt.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2), color='r', width=0.0005, head_width=0.0025)
    plt.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2,variables[i], color='r')
for i in range(len(Z1)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.scatter(Z1[i], Z2[i], c='g', marker='o')
    plt.text(Z1[i]*1.2, Z2[i]*1.2, observations[i], color='b')
plt.figure()


# In[11]:


comps = pd.DataFrame(A,columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True,cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Principal components')

