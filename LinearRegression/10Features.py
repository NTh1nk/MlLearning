#What if we had 10 features for each data point?

#In this case, we are importing StandardScaler, and not matplotlib, as we cant visualise the data.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Each data points, has 10 features.
X = np.array([
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



