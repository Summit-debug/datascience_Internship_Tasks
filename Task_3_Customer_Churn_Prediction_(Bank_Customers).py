#%% md
# # Import Library and Load Dataset
#%%
import pandas as pd

df = pd.read_csv('/content/Churn_Modelling.csv')
#%% md
# ## Clean and prepare Dataset
#%%
print(df.isnull().sum())

#%% md
# # Encode Categorical Columns
#%%
from sklearn.preprocessing import LabelEncoder

# Label encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

# One-hot encode Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

#%% md
# # Split Data into X and y
#%%
X = df.drop('Exited', axis=1)  # Features
y = df['Exited']               # Target

#%% md
# # Train/Test Split
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%% md
# # Train a Classification Model
# 
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Count of customers who exited (1) vs stayed (0)
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)

plt.title('Count of Customers: Exited vs Stayed')
plt.xlabel('Exited (0 = Stayed, 1 = Exited)')
plt.ylabel('Number of Customers')
plt.xticks([0,1], ['Stayed', 'Exited'])
plt.show()

#%%
