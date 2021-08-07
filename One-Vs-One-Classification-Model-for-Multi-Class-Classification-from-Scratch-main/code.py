import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# method to load the data from the internet
def load_data():
    URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header = None)
    print(data)
    
    data = data[:150]
    #print("data type: ", type(data))
    data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
    data = np.asmatrix(data, dtype = 'float64')
    return data

data = load_data()

# data for class 0 and class 1
data_a_b = np.empty((0, 2), float)

#data for class 1 and class 2
data_b_c = np.empty((0, 2), float)

#data for class 2 and class 3
data_a_c = np.empty((0, 2), float)

for i in range(100):
    data_a_b = np.append(data_a_b, data[i])

for i in range(50, 150):
    data_b_c = np.append(data_b_c, data[i])
    
for i in range(0, 50):
    data_a_c = np.append(data_a_c, data[i])

for i in range(100, 150):
    data_a_c = np.append(data_a_c, data[i])

data_a_b = np.resize(data_a_b, (100, 5))
data_b_c = np.resize(data_b_c, (100, 5))
data_a_c = np.resize(data_a_c, (100, 5))

# Plot to show the scatteredness of data
plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
plt.scatter(np.array(data[100:,0]), np.array(data[100:,2]), marker='s', label='virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()


# to check the class based on threshold value
def step_check(v):
    if v >= 0:
        return 1
    else: 
        return 0

# method to learn the proper weights    
def perceptron(data2, num_iter, learning_rate, bias):
    
    #contains the input parameters
    features = np.array([]) 
    
    #contains the weight along with the bias
    w = np.array([bias, 1.0, 1.0])
    
    for i in range(data2.shape[0]):
        a = data2[i, 0]
        b = data2[i, 2]
        jar = np.array([a, b])
        features = np.append(features, jar)
        
    features = np.resize(features, (100, 2))
    
    labels = [0.0] * 50
    for j in range(50):
        labels.append(1.0)
    
    #stores the accuracy rate of the learning algorithm
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
                
        accuracy_rate.append(100 - errors)
        
    return (w, accuracy_rate)
        
#number of iterations for proper learning
num_iter = 2500

#learning to classify between class 0 and class 1
w1, accuracy_rate1 = perceptron(data_a_b, num_iter, 0.00001, 1)

#learning to classify between class 1 and class 2
w2, accuracy_rate2 = perceptron(data_b_c, num_iter, 0.01, 1)

#learning to classify between class 0 and class 2
w3, accuracy_rate3 = perceptron(data_a_c, num_iter, 0.0001, 0.5)


#plotting the accuracy graph
epochs = np.arange(1, num_iter+1)
plt.plot(epochs, accuracy_rate1, linestyle = 'dashed', color = 'red', label ="class 0 and 1")
plt.plot(epochs, accuracy_rate2, linestyle = 'dotted', color = 'green', label ="class 1 and 2")
plt.plot(epochs, accuracy_rate3, color = 'blue', label = "class 0 and 2")
plt.xlabel('Iterations')
plt.ylabel('Accuracy(%age))')
plt.xlim([0, 2500])
plt.legend()
plt.show()

#scatter plot to show how the curve fits
plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa')
plt.scatter(np.array(data[50:,0]), np.array(data[50:,2]), marker='x', label='versicolor')
plt.scatter(np.array(data[100:,0]), np.array(data[100:,2]), marker='s', label='virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.xlim([3, 8])
plt.ylim([0, 7])

x1 = np.linspace(-5, 10, 500)
y1 = (-w1[0] - w1[1]*x1) / w1[2]
plt.plot(x1, y1, color = "red", label = "class 0 and 1")

x2 = np.linspace(-5, 10, 500)
y2 = (-w2[0] - w2[1]*x2) / w2[2]
plt.plot(x2, y2, color = "green", label = "class 1 and 2")


x3 = np.linspace(-5, 10, 500)
y3 = (-w3[0] - w3[1]*x3) / w3[2]
plt.plot(x3, y3, color = "blue", label = "class 0 and 2")

plt.legend()
plt.show()

# testing the data to check the class with respect to the correct labels
def class_decider_method(data, w1, w2, w3, labels):
    
    features = np.array([])
    
    #will store the votes given by different classification algorithms
    output_array = []
    
    for i in range(150):
        output_array.append([0, 0, 0])
    
    for i in range(data.shape[0]):
        a = data[i, 0]
        b = data[i, 2]
        jar = np.array([a, b])
        features = np.append(features, jar)
        
    features = np.resize(features, (150, 2))
    errors = 0
    
    #iterates over each test sample
    for i in range(features.shape[0]):
            v = np.dot(w1[1:], features[i]) + w1[0]
            y = step_check(v)
            
            if (y == 0):
                output_array[i][0] +=1
            else:
                output_array[i][1] += 1
                
            v = np.dot(w2[1:], features[i]) + w2[0]
            y = step_check(v)
            
            if (y == 0):
                output_array[i][1] +=1
            else:
                output_array[i][2] += 1
                
            v = np.dot(w3[1:], features[i]) + w3[0]
            y = step_check(v)
            
            if (y == 0):
                output_array[i][0] +=1
            else:
                output_array[i][2] += 1
            
            inputt = labels[i]
            if (output_array[i][inputt] != 2):
                errors += 1
            
    return errors            

labels = [0] * 50
for i in range(50):
    labels.append(1)
for i in range(50):
    labels.append(2)

errors = class_decider_method(data, w1, w2, w3, labels)

#contains the accuracy of the one vs one algorithm
accuracy_level = ((150 - errors) / 150) * 100.0

print("Accuracy %age: ", accuracy_level)