
### What is logistic regression
Logistic regression is a statistical and machine learning method used for binary classification problems. It is a linear classification algorithm that models the probability of a binary outcome using a logistic function (also known as the sigmoid function). Logistic regression predicts the probability of an event (or class) and classifies it by setting a threshold.
 


###  Mathematical Expression
* The core of logistic regression is the logistic function, which maps the output of a linear equation to the [0, 1] interval. The logistic function is defined as:
$$\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]$$
* where \( z \) is the output of the linear model, given by:
$$\[ z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n \]$$
* Using the logistic function, we convert the linear model's output \( z \) into a probability value:
$$\[ P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}} \]$$
* This probability can be used for classification. Typically, a threshold (such as 0.5) is set: if the predicted probability is greater than or equal to the threshold, the instance is classified as the positive class (1); otherwise, it is classified as the negative class (0).

### Loss Function

* Logistic regression uses maximum likelihood estimation to find the best parameters, which is usually achieved by minimizing the log loss (also known as logistic loss or cross-entropy loss) function:

$$\[ \text{Log Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] \]$$

### Evaluate model performance
* Using the matrix, we can calculate a number of indicators（e.g.，Accuracy,Precision,Recall,F1_Score)
* The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate (1 - Specificity).and AUC measures the entire two-dimensional area underneath the ROC curve. (The higher the AUC value, the better the performance.)


