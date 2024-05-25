# Day 17, 23.05.2024
<span style="color:grey">
Our daily protocol
</span>

---
## <span style="color:black"> __Basic Overview__ </span>
 
* <span style="color:grey"> Presentation Logistic Regression
* <span style="color:grey"> Group work: Logistic Regression


---
##  __Schedule__
<span style="color:grey">

|Time|Content|
|---|---|
|09:30 - 11:05|Presentation: Logistic Regression|
|11:15 - 16:00| Group work: Logistic Regression|



## <span style="color:black"> 1. Presentation: Logistic Regression </span>

### What is logistic regression
Logistic regression is a statistical and machine learning method used for binary classification problems. It is a linear classification algorithm that models the probability of a binary outcome using a logistic function (also known as the sigmoid function). Logistic regression predicts the probability of an event (or class) and classifies it by setting a threshold.
 
 ### Some examples for classification
 * Email: spam vs not spam
 * Tumor classification: malignant vs benign
 * Bee image: healthy and not healthy

### In (binary)-classification we have two possible classes:
* Class 0: the negative class (not spam, malignant, not cat)
* Class 1: the positive class (spam, benign, cat)

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


## <span style="color:black"> 2. Group work: Logistic Regression </span>
[Link to the repo](https://github.com/neuefische/ds-logistic-regression)

* The first two notebooks will show us how logistic regression works graphically and with scikit-learn. 
* In the third notebook, we will find an example of a multiclass classification using one of the most popular data science data sets. 
* In the fourth notebook, it's your time to implement logistic regression on a new data set. I
f you get stuck you can have a look at the solution notebook in the solution branch.
