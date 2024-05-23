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

### : Mathematical Expression
* The core of logistic regression is the logistic function, which maps the output of a linear equation to the [0, 1] interval. The logistic function is defined as:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 

where 
𝑧
z is the output of the linear model, given by:

𝑧
=
𝛽
0
+
𝛽
1
𝑥
1
+
𝛽
2
𝑥
2
+
⋯
+
𝛽
𝑛
𝑥
𝑛
z=β 
0
​
 +β 
1
​
 x 
1
​
 +β 
2
​
 x 
2
​
 +⋯+β 
n
​
 x 
n
​

* Class 1: the positive class (spam, benign, cat)

## <span style="color:black"> 2. Group work: Logistic Regression </span>
[Link to the repo](https://github.com/neuefische/ds-logistic-regression)

* The first two notebooks will show us how logistic regression works graphically and with scikit-learn. 
* In the third notebook, we will find an example of a multiclass classification using one of the most popular data science data sets. 
* In the fourth notebook, it's your time to implement logistic regression on a new data set. I
f you get stuck you can have a look at the solution notebook in the solution branch.
