#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
x = np.random.randint(1, 20, 15)
print("\nRandom vector of size 15 having integers in the range of 1-20 is : \n", x)


reshapedArray = x.reshape((3, 5))
print("\nReshaped 3 by 5 array is : \n", reshapedArray)


shape = np.shape(reshapedArray)
print("\nShape of the array is : ", shape)

maxNum = np.amax(reshapedArray, axis=1)

newArray = np.where(np.isin(reshapedArray, maxNum), 0, reshapedArray)
print("\nArray after replacing the max in each row by 0 is : \n", newArray)


# In[17]:


y = np.array([[2, 4, 6], [6, 8, 10],[8,10,12],[10,12,14]], np.int32)
print(type(y))	
print(y.shape)
print(y.dtype)


# In[18]:


m = np.mat("3 -2;1 0")
print("Original matrix:")
print("a\n", m)
w, v = np.linalg.eig(m) 
print( "Eigenvalues of the said matrix",w)
print( "Eigenvectors of the said matrix",v)


# In[19]:


n_array = np.array([[0,1,2],
                    [3,4,5]])
  
print("Numpy Matrix is:")
print(n_array)
  
# calculating the Trace of a matrix
trace = np.trace(n_array)
  
  
print("\nTrace of given 3X3 matrix:")
print(trace)


# In[20]:


x = np.array([[1, 2],
            [3,4],
            [5,6]])
print(x.shape)
z = np.reshape(x,(2,3))
print("Reshape 2x3:")
print(z)


# In[21]:


from matplotlib import pyplot as plt

language = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

explode = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)

colors = ("blue", "orange", "green", "red", "indigo", "brown")

wedge_properties = {'linewidth': 1, 'edgecolor': "black"}


def autopact(pct):
    return "{:.1f}%".format(pct)


fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts, autotext = ax.pie(popularity,
                                 autopct=lambda pct: autopact(pct),
                                 explode=explode,
                                 labels=language,
                                 shadow=True,
                                 colors=colors,
                                 startangle=140,
                                 wedgeprops=wedge_properties)

plt.setp(autotext, size=8, weight="bold")

plt.show()


# In[ ]:




