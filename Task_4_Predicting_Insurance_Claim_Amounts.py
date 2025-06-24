#%% md
# # Import library and Load Dataset
#%%
import pandas as pd

df = pd.read_csv('/content/insurance.csv')
#%% md
# # Preprocess Data
#%%
# Example using One-Hot Encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

#%%
print(df.isnull().sum())
#%% md
# # Features and Target
#%%
X = df.drop('charges', axis=1)  # features
y = df['charges']               # target variable

#%% md
# # Train Test_split
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%% md
# # Train Linear Regression Model
#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#%% md
# # Evaluate Model
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

#%% md
# # Visualize How BMI, Age, and Smoking Affect Charges
#%%
import matplotlib.pyplot as plt
import seaborn as sns
#%% md
# 
#%%
test_data = X_test.copy()
test_data['charges_actual'] = y_test
test_data['charges_predicted'] = y_pred

plt.figure(figsize=(16, 4))
#%% md
# # BMI vs Charges
#%%
plt.subplot(1, 3, 1)
sns.scatterplot(x=test_data['bmi'], y=test_data['charges_actual'], label='Actual')
sns.lineplot(x=test_data['bmi'], y=test_data['charges_predicted'], color='red', label='Predicted')
plt.title('BMI vs Charges')
#%% md
# # Age vs Charges
#%%
plt.subplot(1, 3, 2)
sns.scatterplot(x=test_data['age'], y=test_data['charges_actual'], label='Actual')
sns.lineplot(x=test_data['age'], y=test_data['charges_predicted'], color='red', label='Predicted')
plt.title('Age vs Charges')
#%% md
# # Smoking Status vs Charges
#%%
plt.subplot(1, 3, 3)
# For smoking, since encoded as binary column 'smoker_yes' (assuming OneHotEncoding)
sns.boxplot(x=df['smoker_yes'], y=df['charges'])
plt.title('Smoking Status vs Charges')
plt.xlabel('Smoker (0=No, 1=Yes)')

plt.tight_layout()
plt.show()
#%%
