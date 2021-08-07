# 10_fold_cross_validation_from_scratch

Code is implemented from scratch.

The code uses the following algorithm:
<ol>
	<li>Shuffle the dataset</li>
	<li>Break the dataset into 10 equal parts</li>
	<li>Use one part as testing data and other parts as training data</li>
	<li>In each of the 10 iterations, calculate the root mean square error</li>
	<li>Average out the root mean square error of all 10 iterations</li>
</ol>

Dataset is used from Kaggle. Click <a href = "https://www.kaggle.com/kennethjohn/housingprice">here</a> for the dataset.