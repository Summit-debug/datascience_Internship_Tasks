#%% md
# ## Import Libraries
#%%
import pandas as pd
#%%

df = pd.read_csv('/content/loan_approval_dataset.csv')  # or whatever your file is called

#%% md
# # checking missing vakues
#%%
# Check missing values
print(df.isnull().sum())


#%%

#%% md
# # Histogram
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
plt.figure(figsize=(7, 4))
sns.histplot(df[' loan_amount'], bins=30, kde=True)
plt.title("Distribution of Loan Amount")
plt.show()
#%%

#%% md
# # Countplot
#%%
import numpy as np
#%%
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x=' education', hue=' loan_term')
plt.title("Loan Status by Education")
plt.show()
#%%
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x=' education', hue=' loan_amount')
plt.title("Loan Status by Education")
plt.show()
#%%
# If using lowercase cleaned column names
df.columns = df.columns.str.strip().str.lower()

# Now this will work
sns.histplot(df['income_annum'], kde=True, bins=30)
plt.title("Applicant Income Distribution")
plt.xlabel("Income (Annum)")
plt.show()

#%% md
# # prepare data for training
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#%%
# 'loan_id' is just an identifier, not useful for training
X = df.drop(columns=['loan_amount', 'loan_status'])  # Features
y = df['loan_status']                            # Target

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%% md
# # Train the model by Logistic Regression
# 
#%%
print(X.dtypes)


#%%
from sklearn.preprocessing import LabelEncoder

# Make a copy to avoid modifying original data
X_encoded = X.copy()

# Encode all object-type columns
le = LabelEncoder()
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        X_encoded[col] = le.fit_transform(X_encoded[col])

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

#%%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#%% md
# # evaluate
#%%
y_pred_lr = lr.predict(X_test)

#%%
# Accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#%%
