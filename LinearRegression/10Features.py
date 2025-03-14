#What if we had 10 features for each data point?

#In this case, we are importing StandardScaler to scale the features, so they are comparable
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# Each data points, has 10 features.
x = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
])

#targert (y) values, i have made it so, the y value is just the sum of the features. With a bit of varience
y = np.array([55, 65, 75, 85, 95, 105, 115, 128, 135, 143])

scaler = StandardScaler() # Scaling
X_scaled = scaler.fit_transform(x) #scaling

model = LinearRegression()
model.fit(X_scaled, y)

#We have now made the model, its tiem for testing


#New data
new_data = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("Prediction", prediction[0])

#The model, is finished, now it is time to take a look into the coeffecients


print("Coefficents", model.coef_) #model.coef_ is how you get the coeffecints
print("Intercept", model.intercept_)


#Now we can test the accuracy using R^2

r2 = r2_score(y, model.predict(X_scaled))

print("R^2 Score:", r2)
#Now we want to visualise the data 
plt.scatter(y, model.predict(X_scaled), color="red", label="Prediction vs Real")
plt.xlabel("Real values (y)")
plt.ylabel("Predicted values")
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", linestyle="dashed", label="Perfect Prediction")
plt.legend()
plt.title("Prediction vs Real")
plt.show()

