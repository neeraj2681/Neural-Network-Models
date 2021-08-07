# One-Vs-One-Classification-Model-for-Multi-Class-Classification-from-Scratch
<b>Classifying multi-class dataset into different classes using binary classification algorithm</b>

<b>NOTE: THIS IS A PURELY LEARNING ALGORITHM AND HENCE SAME DATASET IS USED FOR LEARNING AS WELL AS TESTING.</b>

The dataset is taken from IRIS website. <a href= "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data">Click</a> to see the dataset.

The dataset contains three kinds of flowers <b>Setosa, Versicolor</b> and <b>Virginica</b> with 5 parameters, but in the classification algorithm we'll use only two parameters namely <b>petal length</b>(given in first column) and <b>sepal length</b>(given in the third column) to make the program easier to work with.

This is a purely learning algorithm, so the whole dataset will be used as the learning dataset and after the classification of dataset, the testing will be done on the same dataset. As the dataset is not purely linearly separable, the accuracy even after learning on the whole dataset will not be equal to 100%.

The algorithm uses the <b>one-vs-one</b> model instead of one-vs-rest model. 

In one-vs-one model, for <img src="https://latex.codecogs.com/svg.image?n" title="n" /> class dataset, for each test case the binary classification is done for each  <img src="https://latex.codecogs.com/svg.image?\binom{n}{2}" title="\binom{n}{2}" /> pair of classes and on the basis of the outcome, a vote is casted in the favour of the output class. The final decision is made in favour of the class with maximum votes. In this particular algorithm, if a draw happens for a sample test case(like 1 vote for each class), then the algorithm will declare the output as wrong and the accuracy will fall down.

The program will first print the whole data, then it'll plot the data points to show the density of data.
Then the program will learn using the algoritm and later it'll test the same data.
The accuracy graph and the separation graph for the 3 binary classifications will be shown by the algorithm.

At last the code will print the accuracy of the whole algorithm. 
