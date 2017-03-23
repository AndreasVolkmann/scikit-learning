import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
cols = ["TV", "Radio", "Newspaper"]

sns.pairplot(data, x_vars=cols, y_vars="Sales", size=7, aspect=0.7, kind='reg')
# sns.plt.show()

feature_cols = cols[0:3]
X = data[feature_cols]
y = data.Sales

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print(linreg.intercept_)
for i in zip(feature_cols, linreg.coef_): print(i)

# Predict
y_pred = linreg.predict(X_test)

print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

lr = LinearRegression()
scores = cross_val_score(lr, X, y, cv=10, scoring="mean_squared_error")

# negate scores
mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)
first = rmse_scores.mean()

# excluding Newspaper
feature_cols = cols[0:2]
X = data[feature_cols]
second = np.sqrt(-cross_val_score(lr, X, y, cv=10, scoring="mean_squared_error")).mean()

print(f"With Newspaper: {first}")
print(f"Without: {second}")

