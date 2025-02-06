# -*- coding: utf-8 -*-
"""courseworkDM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Mnvam_OwkFLj6t06SJDMKCGcAHZ4zuMm
"""

import pandas as pd
data = pd.read_csv("abalone.head")
print(data.head())

# I checked the source to understand what the labels are.
# I will add the names just for clarity and to make it easier to understand the code.
# I used the source: https://archive.ics.uci.edu/dataset/1/abalone  to understand what the labels for each column are, and name them respectively.

data.columns = [ "Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]

# Now I'm printing the df data to check if the labels are correct and visualize them.

cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
           'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

empty_values = {}

for col in cols:
    empty_values[col] = data[col].isnull().values.any() or data[col].isna().values.any()


for col, missing_vals in empty_values.items():
    print(f"{col}: {missing_vals}")

# as we can see after running the above checks, there are no empty or missing values in the dataset.

# I will now add an Age column to the data frame which conains the age of the abalone,
# which can be derived from the Rings column, where you add 1.5 to the number of rings.

for row in data.iterrows():
    data['Age'] = data['Rings'] + 1.5

data['Sex'].replace(['M', 'F', 'I'], [0, 1, 2], inplace=True)

print(data.head())

cols_to_standardize = ['Length', 'Diameter', 'Height', 'Whole_weight',
                          'Shucked_weight', 'Viscera_weight', 'Shell_weight',
                          'Rings']


for col in cols_to_standardize:
    stand_col = f"{col}_Standardized"
    data[stand_col] = (data[col] - data[col].mean()) / data[col].std()

print(data.head())

# to calculate the amount of skewness

skewness = data[['Length', 'Height', 'Diameter', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']].skew()

print("Skewness of weight variables:")
print(skewness)

import matplotlib.pyplot as plt
import numpy as np

# Now, I will calculate some summary statistics to get a better understanding of the data and some relationships.
# I will use the pandas library to do this.

# First I will calculate the percentage of each sex category
male_percentage = print(round((data[data['Sex'] == "M"].shape[0])* 100 / data.shape[0]))  # percentage of males in the data
female_percentage = print(round((data[data['Sex'] == "F"].shape[0])* 100 / data.shape[0]))  # percentage of females in the data
infant_percentage = print(round((data[data['Sex'] == "I"].shape[0])* 100 / data.shape[0]))  # percentage of infants in the data

# I will now plot a box plot of the ages of the abalone and their corresponding sex category

data.boxplot(column = 'Age', by='Sex')
plt.title("Boxplot of Age by Sex")
plt.suptitle('')
plt.xlabel("Sex of Abalone")
plt.ylabel("Age of Abalone")
plt.show()

# Based on the boxplot, considering there are a significant amount of outliers, and the median values of the age for each sex are relatively close, I would say that the sex alone does not have a huge affect on the age, especially considering male and female abalones have approximately the same median age.

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

X = data[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
y = data['Age']


num_cols = 4
num_rows = (len(X.columns)+num_cols-1)//num_cols  # To get rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    axes[i].scatter(X[col], y, alpha=0.5)
    axes[i].set_title(f'{col} vs Age')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Age')
    corr, _ = pearsonr(X[col], y)
    axes[i].text(0.05, 0.95, f'Pearson r = {corr:.2f}', transform=axes[i].transAxes,
                 verticalalignment='top')

plt.tight_layout()
plt.show()

# Used for investigating feature engineering


data['Length_Diameter_Mean'] = (data['Length_Standardized'] + data['Diameter_Standardized']) / 2

plt.scatter(data['Length_Diameter_Mean'], data['Age'], alpha=0.5)
plt.title('Length_Diameter_Mean vs Age')
plt.xlabel('Length_Diameter_Mean')
plt.ylabel('Age')
plt.show()




data['Height_Diameter_Sum'] = data['Height_Standardized'] + data['Diameter_Standardized']
data['Log_Height_Diameter_Sum'] = np.log1p(data['Height_Diameter_Sum'])

plt.scatter(data['Height_Diameter_Sum'], data['Age'], alpha=0.6)
plt.title('Height_to_Diameter vs Age')
plt.xlabel('Height_Diameter_Mean')
plt.ylabel('Age')
plt.grid(True)
plt.show()

X = data[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]


plt.figure(figsize=(12, 10))
plt.boxplot([X[col] for col in X.columns], labels=X.columns)

plt.title('Box Plots for Abalone Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

k = 2.5  # Adjusted k for extended IQR
data_cleaned = data.copy()

for col in cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1-k*IQR
    upper_bound = Q3+k*IQR

    data_cleaned = data_cleaned[(data_cleaned[col] >= lower_bound) & (data_cleaned[col] <= upper_bound)]

print(f"Original data size: {data.shape}")
print(f"Cleaned data size: {data_cleaned.shape}")

plt.figure(figsize=(12, 10))
plt.boxplot([data_cleaned[col] for col in cols], labels=cols)
plt.title('Box Plots for Features (Outliers Removed)')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = data[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
y = data['Age']

feature_names = data.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)

rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

importances = rf_regressor.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Gini Importance': importances})
print(feature_imp_df)

print("Lasso Model Score: ", lasso_model.score(X_test, y_test))
lasso_model.coef_

lasso_model = Lasso(alpha=0.01)  # Alpha controls the regularization strength
lasso_model.fit(X_train, y_train)

lasso_coefs = pd.DataFrame({
    'Feature': X.columns,
    'Lasso Coefficient': lasso_model.coef_
})

print(lasso_coefs.sort_values(by='Lasso Coefficient', ascending=True))

# According to the output, the relevant features for determining the age of the abalone is Whole_weight, given the alpha values of 0.1, 0.2, 0.3, 0.4, 0.5.

# https://vitalflux.com/lasso-ridge-regression-explained-with-python-example/
# https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
# https://www.geeksforgeeks.org/feature-importance-with-random-forests/

# Simple Neural Network Model (no tuning of hyper-parameters)

# MAE: 1.37, MSE: 3.96, R²: 0.66 (all samples)

# for k = 3.5  MAE: 1.43, MSE: 4.35, R²: 0.63 (3997)

# for k = 3.  MAE: 1.57, MSE: 5.20, R²: 0.57 (3994)

# for k = 2.5  MAE: 1.57, MSE: 5.20, R²: 0.57 (3989)

# for k = 2.  MAE: 1.49, MSE: 4.96, R²: 0.57 (3964)


# https://keras.io/api/
# https://scikit-learn.org/

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from sklearn.decomposition import PCA
import random
import numpy as np
import tensorflow as tf

X = data[['Sex', 'Length_Standardized', 'Diameter_Standardized', 'Height_Standardized',
          'Whole_weight_Standardized', 'Shucked_weight_Standardized',
          'Viscera_weight_Standardized', 'Shell_weight_Standardized']]
y = data_cleaned['Age']

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}")

# Test Model Using Minkowski Error and Huber Loss

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import random
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt


X = data[['Sex', 'Length_Normalized', 'Diameter_Normalized', 'Height_Normalized',
          'Whole_weight_Normalized', 'Shucked_weight_Normalized',
          'Viscera_weight_Normalized', 'Shell_weight_Normalized']]
y = data['Age']

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.10, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, random_state=seed)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.95

    else:
        return lr

def minkowski_error(y_true, y_pred, p=1.5):
    return tf.reduce_mean(tf.abs(y_true - y_pred) ** p)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1)
])



model.compile(optimizer=Adam(learning_rate=0.01), loss=Huber(delta=2.0), metrics=['mae'])

scheduler = LearningRateScheduler(lr_schedule)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=32, callbacks=[scheduler])

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# General Code to Check Pred vs Actual Values for a Model

import pandas as pd

y_pred = model.predict(X_test).flatten()

comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)
print(comparison)

# General Code to Plot The Predicted Age against the True Age

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('True vs Predicted Age')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

columns = ['Length', 'Diameter', 'Height',
                    'Whole_weight', 'Shucked_weight',
                    'Viscera_weight', 'Shell_weight', 'Age']

correlation_matrix = data[columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f', square=True, linewidths=.5)
plt.title('Correlation Heatmap for Selected Abalone Features')
plt.show()

# TEST FOR AUTOENCODERS

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X = data[['Length_Standardized', 'Diameter_Standardized', 'Height_Standardized',
          'Whole_weight_Standardized', 'Shucked_weight_Standardized',
          'Viscera_weight_Standardized', 'Shell_weight_Standardized']]
y = data['Age']

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.10, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, random_state=seed)

input_dim = X_train.shape[1]
encoding_dim = 3

input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
encoder = Dropout(0.2)(encoder)
encoder = Dense(32, activation='relu')(encoder)
latent_space = Dense(encoding_dim, activation='relu', name="latent_space")(encoder)

decoder = Dense(32, activation='relu')(latent_space)
decoder = Dense(64, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=100, batch_size=32, verbose=1)

encoder_model = Model(inputs=input_layer, outputs=latent_space)

X_train_encoded = encoder_model.predict(X_train)
X_val_encoded = encoder_model.predict(X_val)
X_test_encoded = encoder_model.predict(X_test)

X_train_combined = np.hstack([X_train, X_train_encoded])
X_val_combined = np.hstack([X_val, X_val_encoded])
X_test_combined = np.hstack([X_test, X_test_encoded])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

regressor = Sequential([
    Input(shape=(X_train_combined.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_regressor = regressor.fit(X_train_combined, y_train, validation_data=(X_val_combined, y_val), epochs=100, batch_size=32, verbose=1)

y_pred = regressor.predict(X_test_combined).flatten()
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {MAE:.2f}")
print(f"Mean Squared Error: {MSE:.2f}")
print(f"R-squared Score: {R2:.2f}")

plt.plot(history.history['loss'], label='Training Loss (Autoencoder)')
plt.plot(history.history['val_loss'], label='Validation Loss (Autoencoder)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Autoencoder Training and Validation Loss')
plt.show()
plt.plot(history_regressor.history['loss'], label='Training Loss (Regressor)')
plt.plot(history_regressor.history['val_loss'], label='Validation Loss (Regressor)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Regressor Training and Validation Loss')
plt.show()