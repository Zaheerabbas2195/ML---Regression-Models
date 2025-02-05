# -*- coding: utf-8 -*-

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting data into Training set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

# Feature Scalling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""
# Fitting the simple Linear Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# predicting the new results
y_pred = regressor.predict(np.array([[6.5]]))

# Visualizing
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(-1, 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Decision regression')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()


