import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model	


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s

def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_h = 3
	n_y = Y.shape[0]

	return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros((n_y, 1))

	assert (W1.shape == (n_h, n_x))
	assert (b1.shape == (n_h, 1))
	assert (W2.shape == (n_y, n_h))
	assert (b2.shape == (n_y, 1))

	parameters = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}

	return parameters

def forward_propagation(X, parameters):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)

	assert(A2.shape == (1, X.shape[1]))

	cache = {
		"Z1": Z1,
		"A1": A1,
		"Z2": Z2,
		"A2": A2
	}

	return A2, cache

def compute_cost(A2, Y, parameters):
	m = Y.shape[1]

	logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
	cost = -1/m * np.sum(logprobs)

	cost = np.squeeze(cost)

	assert(isinstance(cost, float))

	return cost

def backward_propagation(X, Y, cache, parameters):
	m = X.shape[1]

	A1 = cache["A1"]
	A2 = cache["A2"]

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	dZ2 = A2 - Y
	dW2 = 1/m * np.dot(dZ2, A1.T)
	db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = 1/m * np.dot(dZ1, X.T)
	db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

	grads = {
		"dW1": dW1,
		"db1": db1,
		"dW2": dW2,
		"db2": db2
	}

	return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}

	return parameters	

def predict(X, parameters):
	A2, cache = forward_propagation(X, parameters)
	predictions = (A2 > 0.5)

	return predictions

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
	np.random.seed(3)
	n_x = layer_sizes(X, Y)[0]
	n_y = layer_sizes(X, Y)[2]

	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	for i in range(num_iterations):
		A2, cache = forward_propagation(X, parameters)

		cost = compute_cost(A2, Y, parameters)

		grads = backward_propagation(X, Y, cache, parameters)

		parameters = update_parameters(parameters, grads)

		if print_cost and i % 1000 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	return parameters


np.random.seed(1)
m = 400
N = int(m/2)
D = 2
X = np.zeros((m, D))
Y = np.zeros((m, 1), dtype='uint8')

X, Y = sklearn.datasets.make_moons(n_samples = N, noise = 0.20)

X = X.T
Y = Y.reshape(1, Y.shape[0])

n_h = 5
parameters = nn_model(X, Y, n_h, num_iterations = 5000, print_cost = True)
plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
predictions = predict(X, parameters)
accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T)) / float(Y.size) * 100)
print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
#plt.title("Decision Boundary for hidden layer size 3")
plt.show()