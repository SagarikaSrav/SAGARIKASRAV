#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Apply PCA on CC General data set
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
cc_data = pd.read_csv('CC GENERAL.csv')

# Drop the categorical columns and ID column if present
cc_data = cc_data.drop(['CUST_ID', 'TENURE'], axis=1)

# Fill any missing values with mean of respective column
cc_data = cc_data.fillna(cc_data.mean())

# Scale the data using StandardScaler
scaler = StandardScaler()
cc_scaled = scaler.fit_transform(cc_data)

# Initialize PCA model with 2 components
pca = PCA(n_components=2)

# Fit and transform the data using PCA
cc_pca = pca.fit_transform(cc_scaled)

# Print the explained variance ratio
print('Explained variance ratio:', pca.explained_variance_ratio_)

# Create a new dataframe with the transformed data
cc_pca_df = pd.DataFrame(data=cc_pca, columns=['PC1', 'PC2'])

# Print the transformed data


# In[11]:


#Calculate silhouette score without applying pca
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
cc_data = pd.read_csv('CC GENERAL.csv')

# Drop the categorical columns and ID column if present
cc_data = cc_data.drop(['CUST_ID', 'TENURE'], axis=1)

# Fill any missing values with mean of respective column
cc_data = cc_data.fillna(cc_data.mean())

# Scale the data using StandardScaler
scaler = StandardScaler()
cc_scaled = scaler.fit_transform(cc_data)

# Initialize k-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the k-means model on the scaled data
kmeans.fit(cc_scaled)

# Calculate the silhouette score of the clustered data
silhouette_avg = silhouette_score(cc_scaled, kmeans.labels_)

# Print the silhouette score
print('Silhouette score:', silhouette_avg)


# In[12]:


#Calculate silhouette score applying pca
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
cc_data = pd.read_csv('CC GENERAL.csv')

# Drop the categorical columns and ID column if present
cc_data = cc_data.drop(['CUST_ID', 'TENURE'], axis=1)

# Fill any missing values with mean of respective column
cc_data = cc_data.fillna(cc_data.mean())

# Scale the data using StandardScaler
scaler = StandardScaler()
cc_scaled = scaler.fit_transform(cc_data)

# Initialize PCA model with 2 components
pca = PCA(n_components=2)

# Fit and transform the data using PCA
cc_pca = pca.fit_transform(cc_scaled)

# Initialize k-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the k-means model on the PCA transformed data
kmeans.fit(cc_pca)

# Calculate the silhouette score of the clustered data
silhouette_avg = silhouette_score(cc_pca, kmeans.labels_)

# Print the silhouette score
print('Silhouette score:', silhouette_avg)


# In[13]:


#Perform Scaling+PCA+K-Means and report performance with 2 clusters in kmeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
cc_data = pd.read_csv('CC GENERAL.csv')

# Drop the categorical columns and ID column if present
cc_data = cc_data.drop(['CUST_ID', 'TENURE'], axis=1)

# Fill any missing values with mean of respective column
cc_data = cc_data.fillna(cc_data.mean())

# Scale the data using StandardScaler
scaler = StandardScaler()
cc_scaled = scaler.fit_transform(cc_data)

# Apply PCA
pca = PCA(n_components=2)
cc_pca = pca.fit_transform(cc_scaled)

# Initialize k-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the k-means model on the PCA data
kmeans.fit(cc_pca)

# Calculate the silhouette score of the clustered data
silhouette_avg = silhouette_score(cc_pca, kmeans.labels_)

# Print the silhouette score
print('Silhouette score:', silhouette_avg)


# In[14]:


#Perform Scaling+PCA+K-Means and report performance with 3 clusters in kmeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
cc_data = pd.read_csv('CC GENERAL.csv')

# Drop the categorical columns and ID column if present
cc_data = cc_data.drop(['CUST_ID', 'TENURE'], axis=1)

# Fill any missing values with mean of respective column
cc_data = cc_data.fillna(cc_data.mean())

# Scale the data using StandardScaler
scaler = StandardScaler()
cc_scaled = scaler.fit_transform(cc_data)

# Apply PCA
pca = PCA(n_components=2)
cc_pca = pca.fit_transform(cc_scaled)

# Initialize k-means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the k-means model on the PCA data
kmeans.fit(cc_pca)

# Calculate the silhouette score of the clustered data
silhouette_avg = silhouette_score(cc_pca, kmeans.labels_)

# Print the silhouette score
print('Silhouette score:', silhouette_avg)


# In[15]:


#Use pd_speech_features.csv perform scaling
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('pd_speech_features.csv')

# Split the data into features and target variable
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a new dataframe with the standardized features
std_df = pd.DataFrame(X_std, columns=X.columns)

# Add the target variable to the new dataframe
std_df['Target'] = y

# Visualize the standardized data
print(std_df.head())


# In[16]:


#Use pd_speech_features.csv to Apply PCA (k=3)
# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('pd_speech_features.csv')

# Split the data into features and target variable
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object with 3 principal components
pca = PCA(n_components=3)

# Fit the PCA model on the standardized data
pca.fit(X_std)

# Transform the data to the new coordinate system
X_pca = pca.transform(X_std)

# Visualize the explained variance ratio of the principal components
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create a new dataframe with the transformed data
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])

# Add the target variable to the new dataframe
pca_df['Target'] = y

# Visualize the transformed data
print(pca_df.head())


# In[17]:


#Use pd_speech_features.csv to Use SVM to report performance 
# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('pd_speech_features.csv')

# Split the data into features and target variable
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a PCA object with 3 principal components
pca = PCA(n_components=3)

# Fit the PCA model on the standardized data
pca.fit(X_std)

# Transform the data to the new coordinate system
X_pca = pca.transform(X_std)

# Split the transformed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Create an SVM object with a linear kernel
svm = SVC(kernel='linear', random_state=42)

# Fit the SVM model on the training data
svm.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = svm.predict(X_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy:", accuracy)


# In[19]:


#3Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data tok=2.
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the dataset
df = pd.read_csv('Iris.csv')

# Split the data into features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert the target variable to numeric format
y = pd.factorize(df.iloc[:, -1])[0]

# Create an LDA object with 2 discriminant components
lda = LinearDiscriminantAnalysis(n_components=2)

# Fit the LDA model on the data and transform the data to the new coordinate system
X_lda = lda.fit_transform(X, y)

# Plot the transformed data in a scatter plot, coloring points by target variable
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.xlabel('Discriminant Component 1')
plt.ylabel('Discriminant Component 2')
plt.show()


# In[20]:


#4. Briefly identify the difference between PCA and LDA using Iris data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with k=2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply LDA with k=2
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(121)
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.legend()

plt.subplot(122)
for i, target_name in enumerate(target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=target_name)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA')
plt.legend()

plt.show()


# In[ ]:




