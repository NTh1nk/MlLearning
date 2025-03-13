#Import depencies#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
#Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)  
y = np.array([2.2, 4.7, 6.4, 8.9, 10.7, 12.2, 14, 16, 18])  
#Model, in this case LinearRegression
model = LinearRegression()
#fit the model
model.fit(X, y)
#Predict a num
next_number = model.predict([[10]])


print("Predicted next number:", next_number[0])
print("Slope/Hældning (m/a):", model.coef_[0])
print("Intercept/Skæringspunktet (b):", model.intercept_)
