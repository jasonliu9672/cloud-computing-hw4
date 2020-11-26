import numpy as np

raw = np.loadtxt('data_banknote_authentication.txt',delimiter=',')
X = np.array([item[0:4] for item in raw])
y = np.array([item[4] for item in raw])
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def gradient(X, y, w):
    g = np.zeros(len(w))
    for x,y in zip(X, y):
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        g += (error - y) * x
    return g / len(X)

def cost(X, y, w):
    total_cost = 0
    for x,y in zip(X, y):
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        total_cost += abs(y - error)
    return total_cost

def logistic(X,y):
    w = np.zeros(4)
    limit = 50 
    eta = 0.1 
    costs = []
    for i in range(limit):
        current_cost = cost(X, y, w)
        print "current_cost=",current_cost
        costs.append(current_cost)
        w = w - eta * gradient(X, y, w)
        eta *= 0.95
    return w

def predict(x, w):
    score = 0
    for i in range(len(x)):
        score += w[i] * x[i]
    return sigmoid(score)

def evaluate():
    predictions = []
    error = 0
    for x in X:
        predict_y = round(predict(x, w))
        predictions.append(predict_y)
    for i in range(len(y)):
        if y[i] != predictions[i]:
            error += 1

    return error/ float(len(y)) * 100.0

w = logistic(X,y)
print('Training Error: %.3f%%' % evaluate())

