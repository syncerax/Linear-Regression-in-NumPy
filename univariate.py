from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt('ex1data1.txt', delimiter=',')
m = len(data)

plt.title("Population and Profits")
plt.xlabel("Population (in 10000s)")
plt.ylabel("Profits (in $10000s)")
plt.plot(data[:, 0], data[:, 1], 'rx')
plt.show()

y = data[:, -1]
X = np.ones((m, data.shape[1]))
X[:, 1:] = data[:,0:-1]

theta = np.zeros(X.shape[1])
iterations = 1500
alpha = 0.01

def compute_cost(X, y, theta):
    m = len(y)
    return sum((X.dot(theta) - y) ** 2) / (2 * m)

J = compute_cost(X, y, theta)
print('With theta = [0, 0]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 32.07\n')
J = compute_cost(X, y, [-1, 2])
print('With theta = [-1, 2]\nCost computed = {}'.format(J))
print('Expected cost value (approx) 54.24\n')

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)
    for i in range(iterations):
        # delta = (1 / m) * sum(((X.dot(theta) - y) * X.T).T)
        delta = (1 / m) * ((X.dot(theta) - y).dot(X))
        theta = theta - alpha * delta
        new_cost = compute_cost(X, y, theta)
        # print("{}. theta = {}, cost = {}".format(i, theta, new_cost))
        J_history[i] = new_cost;
    return theta, J_history

print('Running gradient descent...')
theta, J_history = gradient_descent(X, y, theta, alpha, iterations);

print('Theta found by gradient descent = {}'.format(theta))
print('Expected theta values (approx) = [-3.6303, 1.1664]')

x = np.array([1, 3.5])
prediction = x.dot(theta) * 10000
print('For:\npopulation = 35,000, we predict\nprofit = {}\n'.format(prediction))

x = np.array([1, 7])
prediction = x.dot(theta) * 10000
print('For:\npopulation = 70,000, we predict\nprofit = {}'.format(prediction))

points = np.array([
    [1, X[:, 1].min()], 
    [1, X[:, 1].max()]
])

plt.title("Population and Profits")
plt.xlabel("Population (in 10000s)")
plt.ylabel("Profits (in $10000s)")
plt.plot(data[:, 0], data[:, 1], 'rx')
plt.plot(points[:, 1], points @ theta)
plt.show()