import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading and loading the data
def load_data():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)
    print(data)

# make the dataset linearly separable
    data = data[:100]
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data

data = load_data()


#plotting the scatter graph
plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()

#checking the class based on threshold
def step_check(v):
    if v >= 0:
        return 1
    else: 
        return 0

    
def perceptron(data, num_iter, learning_rate, bias):
    
    features = np.array([])
    w = np.array([bias, 1.0, 1.0])

    for i in range(data.shape[0]):
        a = data[i, 0]
        b = data[i, 2]
        jar = np.array([a, b])
        features = np.append(features, jar)
        
    features = np.resize(features, (100, 2))
    labels = [0.0] * 50
    for j in range(50):
        labels.append(1.0)
    
    
    
    accuracy_rate = [] 
      
    for epoch in range(num_iter):
        errors = 0
        
        for i in range(features.shape[0]):
            v = np.dot(w[1:], features[i]) + w[0]
            y = step_check(v)
            if(y < labels[i]): 
                errors += 1
                w[1:] += (learning_rate * features[i])
                w[0] = w[0] + learning_rate
            elif y > labels[i]:
                errors += 1
                w[1:] = w[1:] - learning_rate * features[i]
                w[0] = w[0] - learning_rate
        accuracy_rate.append(100.0 - errors)
    return (w, accuracy_rate)
             
num_iter = 1000

w, accuracy_rate = perceptron(data, num_iter, 0.0001, 0.5)
#plotting the graph
epochs = np.arange(1, num_iter+1)
plt.plot(epochs, accuracy_rate, linestyle = 'dashed', label = "class 0 and 1")
plt.xlabel('Iterations')
plt.ylabel('Accuracy %age')
plt.xlim([0, 200])
plt.legend()
plt.show()

#scatter plot
plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.xlim([3, 8])
plt.ylim([0, 7])
x = np.linspace(-5, 10, 500)
y = (-w[0] - w[1]*x) / w[2]
plt.plot(x, y, color = "red", label = "class 0 and 1")
plt.legend()
plt.show()