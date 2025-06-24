#%% md
# Import library and Load Dataset
#%%
import pandas as pd

df = pd.read_csv('/content/bank-full.csv', sep=';')  # The delimiter is often `;` in UCI datasets
df.head()

#%%
print(df.info())
print(df['job'].value_counts())
print(df['marital'].value_counts())
print(df['age'].describe())

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Age distribution
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Job distribution
sns.countplot(y='job', data=df)
plt.title('Job Categories')
plt.show()

# Marital status
sns.countplot(x='marital', data=df)
plt.title('Marital Status')
plt.show()

#%%
df_encoded = pd.get_dummies(df, drop_first=True)

#%%
X = df_encoded.drop('y_yes', axis=1)  # 'y' is target column (yes/no), becomes 'y_yes' after encoding
y = df_encoded['y_yes']

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#%%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


#%%
import numpy as np


#%%
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Feature importance
feature_importance = pd.Series(dt.feature_importances_, index=X.columns)
important_features = feature_importance.sort_values(ascending=False).head(10)

# Plot
important_features.plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.title('Top 10 Important Features')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.show()

#%%
