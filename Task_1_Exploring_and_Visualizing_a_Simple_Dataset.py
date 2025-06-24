#%% md
# ## Import Libraries
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% md
# # Load Dataset
#%%
df = pd.read_csv('/content/iris.csv')

#%% md
# # Display Shape Columns Head
#%%
# Dataset shape
print("Shape of dataset:", df.shape)

# Column names
print("\nColumns in dataset:", df.columns.tolist())

print(df.head())

#%% md
# # Visualization
#%% md
# ## Scatter plot
#%%
import matplotlib.pyplot as plt
import seaborn as sns
#%%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title("Sepal Length vs Petal Length")
plt.show()

#%% md
# # Histogram
#%%
plt.figure(figsize=(8, 5))
sns.histplot(df['petal_width'], kde=True, bins=20)
plt.title("Distribution of Petal Width")
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.show()

#%% md
# # Box plot
#%%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot of Iris Features")
plt.xticks(rotation=45)
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=df)
plt.title("Petal Length by Species")
plt.show()

#%%
