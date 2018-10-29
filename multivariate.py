from matplotlib import pyplot as plt
import numpy as np

# Gradient Descent
data = np.loadtxt('ex1data2.txt', delimiter=',')
m = len(data)

y = data[:, -1]
X = data[:, 0:-1]

print("First 10 examples of the training set:")
print("X:")
print(X[0:10])
print("y:")
print(y[0:10])

def feature_normalize(X):
    # axis=0 gives column wise mean/std. axis=1 gives row wise mean/std.
    # Not specifiying axis returns the mean/std of all elements.
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma
    return X, mu, sigma

X, mu, sigma = feature_normalize(X)

# temp_X = np.ones((m, X.shape[1] + 1))
# temp_X[:, 1:] = X
# X = temp_X
# del(temp_X)

X = np.hstack([np.ones((m, 1)), X])

alpha = 1
iterations = 400
theta = np.zeros(X.shape[1])

def compute_cost(X, y, theta):
    m = len(y)
    return sum((X.dot(theta) - y) ** 2) / (2 * m)

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)
    for i in range(iterations):
        # delta = (1 / m) * sum(((X.dot(theta) - y) * X.T).T)
        # delta = (1 / m) * np.sum(X.T * (h - y), axis=1)
        delta = (1 / m) * ((X.dot(theta) - y).dot(X))
        theta = theta - alpha * delta
        new_cost = compute_cost(X, y, theta)
        # print("{}. theta = {}, cost = {}".format(i, theta, new_cost))
        J_history[i] = new_cost;
    return theta, J_history

print('Running gradient descent...')
theta, J_history = gradient_descent(X, y, theta, alpha, iterations);

print('Theta found by gradient descent = {}'.format(theta))

# Predicting the price of a 1650 sqft. 3 bedroom house.
x = np.array([1650, 3])
x = (x - mu) / sigma
x = np.hstack([1, x])
print("Predicted price of a 3 bedroom, 1650 sqft. house (using gradient descent) = $", x.dot(theta))
plt.title("Cost vs Iterations")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.plot(J_history)
plt.show()

# Normal Equation

data = np.loadtxt('ex1data2.txt', delimiter=',')
m = len(data)

y = data[:, -1]
X = np.ones((m, data.shape[1]))
X[:, 1:] = data[:, 0:-1]

def normal_equation(X, y):
    # theta = inv(X.T * X) * X.T * y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

theta = normal_equation(X, y)

x = np.array([1, 1650, 3])
print("Predicted price of a 3 bedroom, 1650 sqft. house (using normal equation) = $", x.dot(theta))