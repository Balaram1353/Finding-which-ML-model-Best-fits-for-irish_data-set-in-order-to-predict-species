#import all the required libraries in oreder to visualize, preprocessing and predicting using machine learning models 
from sklearn.datasets import load_iris #imported the irish dataset from sklearn library 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.show()
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.show()

correlation = df['petal length (cm)'].corr(df['petal width (cm)'])# checking correlation between petal_length and petal_width
print('correlation for petal length and petal width: ', correlation)
correlation = df['sepal length (cm)'].corr(df['sepal width (cm)'])# checking correlation between sepal_length and sepal_width
print('correlation for sepal length and sepal width: ', correlation)

#spliting the data 80% of the data is for training and 20% of data is for analysis
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y = model.predict(X_test)
y.round().astype(int)
mse = mean_squared_error(y_test, y)
print(f"Mean Squared Error for Linear Regression model: {mse}")

print(y_train.shape)
y_train = y_train.squeeze()# converting 2_dimensiomal numpy array in to 1_dimensional numpy array
print(y_train.shape)

scaler = StandardScaler(with_mean=False)
x_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train)
y = model.predict(X_test)
accuracy = accuracy_score(y_test, y)
print(f"accuracy: {accuracy}")

model = RandomForestClassifier()
model.fit(X_train, y_train)
y = model.predict(X_test)
accuracy = accuracy_score(y_test, y)
print(f"accuracy: {accuracy}")

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y = model.predict(X_test)
accuracy = accuracy_score(y_test, y)
print(f"accuracy: {accuracy}")# here KNN model accuracy is 1.0 which means it can be able to predict species 100% with out any errors and other models accuracy is varying from 70 to 80%
