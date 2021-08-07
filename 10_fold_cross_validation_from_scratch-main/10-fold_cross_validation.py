import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd

#method for gradient descent
def gradient_descent(training_data, test_data, epoch):
    alpha = beta = gamma = 0
    n = training_data.shape[0]
    learning_rate = 0.01
    epoch = 1000
    training_flat_size = training_data[:, 0]

    #scaling the flat_size
    training_flat_size = (training_flat_size - np.mean(training_flat_size)) / (
        np.max(training_flat_size) - np.min(training_flat_size))

    training_bedrooms = training_data[:, 1]
    training_prices = np.array(training_data[:, 2])

    #scaling the training prices
    training_prices = (training_prices - np.mean(training_prices)) / (
        np.max(training_prices) - np.min(training_prices))

    test_flat_size = test_data[:, 0]

    #scaling the test flat size
    test_flat_size = (test_flat_size - np.mean(test_flat_size)) / (
                np.max(test_flat_size) - np.min(test_flat_size))
    test_bedrooms = test_data[:, 1]

    test_prices = test_data[:, 2]

    #scaling the test prices
    test_prices = (test_prices - np.mean(test_prices)) / (
        np.max(test_prices) - np.min(test_prices))

    error = 0
    mean_squared_error = 0
    
    for i in range(epoch):
        price_predicted = alpha + beta * training_flat_size + gamma * training_bedrooms
        error = 0
        for j in (training_prices - price_predicted):
            error += j ** 2;
        error = error / n

        alpha_derivative = -(2 / n) * sum(training_prices - price_predicted)
        beta_derivative = -(2 / n) * sum(training_flat_size * (training_prices - price_predicted))
        gamma_derivative = -(2 / n) * sum(training_bedrooms * (training_prices - price_predicted))

        alpha = alpha - learning_rate * alpha_derivative
        beta = beta - learning_rate * beta_derivative
        gamma = gamma - learning_rate * gamma_derivative
    
    for i in range(test_data.shape[0]):
        estimated_price = alpha + beta * test_flat_size[i] + gamma * test_bedrooms[i]
        mean_squared_error += (estimated_price - test_prices[i]) ** 2
        
    return mean_squared_error / test_data.shape[0]

#Kaggle dataset link
data = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/3234/5306/ex1data2.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210517%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210517T045954Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=50e2fac50234ddf192e126c2edce4c0af9be13a25b0f9db6e723756ba870c0909bb0ad0d3a213b21b52b1cfb4dfb94160377f558a591de5c0ee54b916756b2568c02613ee1fffdde5aef0a8c567052821c92093061ff40628026012f4f86d2564293d09d246a57d99f95ba5189f45db6a46d2d92d59073122122029026ff92eef9e76209a871daa7009e8ca037f0860abc0a62cfb70aed8b76c9df29fd237831fece62867012c6ff49cc4f8370925b15c4f8f8a07ea99093cd204404260569865745f5ea95316ed52a600d95fba90704f923c2fd87469a2b3b1d494bec9b3b817cf6e31c1bb89bad7f27ec6c9a97d244d84aca99dfb8aee8dc2ddb93f4b115c3", header = None)
data = data.to_numpy()
np.random.shuffle(data)

#to sum up the mean square error in each fold
total_sum = 0

#to store the number of folds
p = 10

for k in range(0,data.shape[0], data.shape[0] // p):
    test_data = data[k:min(k+5, data.shape[0]), :]
    training_data = np.append(data[:k, :], data[min(k+5, data.shape[0]):, :])
    training_data = np.resize(training_data, (training_data.size // 3, 3))
    total_sum += gradient_descent(training_data, test_data, 1000)
    
print("Average mean squared error: ", total_sum / p)









