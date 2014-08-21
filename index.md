---

title       : Machine Learning with R - Part II
author      : Ilan Man
job         : Strategy Operations  @ Squarespace
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : mathjax       # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}

----

## Agenda 
<space>

1. Logistic Regression
2. Principle Component Analysis
3. Clustering
4. Trees

----

## Objectives 
<space>

1. Understand some popular algorithms and techniques
2. Learn how to tune parameters
3. Practice R

----

## Logistic Regression
# Motivation
<space>

- Describes how a binary response variable (Y = 1 or Y = 0) is associated a set of explanatory variables
- Like Linear Regression, but the outcome is categorical, not continuous
- Instead of:<br>
$\hat{y} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$
- We have:<br>
$\log{\frac{P(Y=1)}{1 - P(Y=1)}} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$
- Can be extended to multiple and/or ordered categories

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_bad_fit](figure/log_bad_fit.png) 

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_bad_fit2](figure/log_bad_fit2.png) 

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_bad_fit3](figure/log_bad_fit3.png) 

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_motivation](figure/log_motivation.png) 

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_motivation2](figure/log_motivation2.png) 

----

## Logistic Regression
# Motivation
<space>

![plot of chunk log_motivation3](figure/log_motivation3.png) 

----

## Logistic Regression
# Concepts
<space>

- Type of regression to predict the probability of being in a class
  - Typical to set threshold to 0.5
- Assumes error terms are Binomially distributed
  - Which generates 1's and 0's as the error term
- Sigmoid or logistic function: $g(z) = \frac{1}{1+e^{-z}}$
  - Interpret the output as $P(Y=1 | X)$
  - Bounded by 0 and 1

----

## Logistic Regression
# Notation
<space>

![plot of chunk log_curve](figure/log_curve.png) 

----

## Logistic Regression
# Find parameters
<space>

- The hypothesis function, $h_{\theta}(x)$, is $P(Y=1|X)$
- Linear regression: $h_{\theta}(x) = \theta x^{T}$
- Logistic regression: $h_{\theta}(x) = g(\theta x^{T})$ 
<br>
where $g(z) = \frac{1}{1+e^{-z}}$

----

## Logistic Regression
# Notation
<space>

- Re-arranging $Y = \frac{1}{1+e^{-\theta x^{T}}}$ yields
<br>
$\log{\frac{Y}{1 - Y}} = \theta x^{T}$
- *Log odds* are linear in $X$
- This is called the logit of $Y$
  - Links $X$ linearly with some function of $Y$
  - When $x_{1}$ increases by 1 unit, $P(Y=1)$ increases by $e^{\theta_{1}}$

----

## Logistic Regression
# Find parameters
<space>

- So $h_{\theta}(x) = \frac{1}{1+e^{-\theta x^{T}}}$
- Cost function?
- Why can't we use the same cost function as for the linear hypothesis?

----

## Logistic Regression
# Find parameters
<space>

- So $h_{\theta}(x) = \frac{1}{1+e^{-\theta x^{T}}}$
- Cost function?
- Why can't we use the same cost function as for the linear hypothesis?
  - Logistic residuals are Binomially distributed
  - Regression function is not linear in $X$

----

## Logistic Regression
# Find parameters
<space>

- Define logistic cost function as:

$cost(h_{\theta}(x)):$<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $= -\log(x)$, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $y = 1$<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $= -\log(1-x)$, &nbsp;   $y = 0$

![plot of chunk cost_curves](figure/cost_curves1.png) ![plot of chunk cost_curves](figure/cost_curves2.png) 

----

## Logistic Regression
# Find parameters
<space>

- Using statistics, it can be shown that<br>
$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>

----

## Logistic Regression
# Find parameters
<space>

- Using statistics, it can be shown that<br>
$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>
- Logistic regression cost function is then<br>
$cost(h_{\theta}(x), y)  = \frac{1}{m} \sum_{i=1}^{m} -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$

----

## Logistic Regression
# Find parameters
<space>

- Using statistics, it can be shown that<br>
<br>
$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>
- Logistic regression cost function is then<br>
<br>
$cost(h_{\theta}(x), y)  = \frac{1}{m} \sum_{i=1}^{m} -y \log(h_{\theta}(x_{i})) + (1-y) \log(1-h_{\theta}(x_{i}))$
- Minimize the cost

----

## Logistic Regression
# Find parameters
<space>

- Cannot solve analytically
- Use approximation methods
  - (Stochastic) Gradient Descent
  - Conjugate Descent
  - Newton-Raphson Method
  - BFGS

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Efficient
- Easier to calculate that gradient descent
  - Except for first and second derivatives
- Converges on *global* minimum

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume $f'(x_{0})$ is zero and $f''(x_{0})$ is positive

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume of $f'(x_{0})$ is zero and $f''(x_{0})$ is positive
- Re-write $f(x)$ as its Taylor expansion:<br>
$f(x) = f(x_{0}) + (x-x_{0})f'(x_{0}) + \frac{1}{2}(x-x_{0})^{2}f''(x_{0})$

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume of $f'(x_{0})$ is zero and $f''(x_{0})$ is positive
- Re-write $f(x)$ as its Taylor expansion:<br>
$f(x) = f(x_{0}) + (x-x_{0})f'(x_{0}) + \frac{1}{2}(x-x_{0})^{2}f''(x_{0})$
- Take the derivative w.r.t $x$ and set = 0<br>
$0 = f'(x_{0}) + \frac{1}{2}f''(x_{0})2(x_{1} − x_{0})$<br>
$x_{1} = x_{0} − \frac{f'(x_{0})}{f￼''(x_{0})}$
  - $x_{1}$ is a better approximation for the minimum than $x_{0}$
  - and so on...

----

## Logistic Regression
# Newton-Raphson Method
<space>

$f(x) = x^{4} - 3\log(x)$

![plot of chunk newton_curve](figure/newton_curve.png) 

----

## Logistic Regression
# Newton-Raphson Method
<space>


```r
fn <- function(x) x^4 - 3*log(x)
dfn <- function(x) 4*x^3 - 3/x
d2fn <- function(x) 12*x^2 + 3/x^2 

newton <- function(num.its, dfn, d2fn){
  theta <- rep(0,num.its)
  theta[1] <- round(runif(1,0,100),0)

  for (i in 2:num.its) {
    h <- - dfn(theta[i-1]) / d2fn(theta[i-1])
    theta[i] <- theta[i-1] + h 
  }
  
  out <- cbind(1:num.its,theta)
  dimnames(out)[[2]] <- c("iteration","estimate")
  return(out)
}
```

----

## Logistic Regression
# Newton-Raphson Method
<space>


```
     iteration estimate
[1,]         1   15.000
[2,]         2   10.000
[3,]         3    6.667
[4,]         4    4.446
[5,]         5    2.968
```

```
      iteration estimate
[16,]        16   0.9306
[17,]        17   0.9306
[18,]        18   0.9306
[19,]        19   0.9306
[20,]        20   0.9306
```

```
[1] 0.9658
```

----

## Logistic Regression
# Newton-Raphson Method
<space>


```r
optimize(fn,c(-100,100))  ## built-in R optimization function
```

```
$minimum
[1] 0.9306

$objective
[1] 0.9658
```

----

## Logistic Regression
# Newton-Raphson - summary
<space>

- Minimization algorithm
- Approximation, non-closed form solution

----

## Logistic Regression
# Summary
<space>

- Very popular classification algorithm
- Part of family of GLMs
- Based on Binomial error terms, i.e. 1's and 0's
- Usually requires large sample size
- Assumes linearity between logit function and independent variables
- Does not work out of the box with correlated features...

----

## Principle Component Analysis
# Motivation
<space>

- Unsupervised learning
- Used widely in modern data analysis
- Compute the most meaningful way to re-express noisy data, revealing the hidden structure
- Commonly used to supplement supervised learning algorithms

----

## Principle Component Analysis
# Concepts
<space>

- Assumes linearity
- $\bf{PX}=\bf{Y}$
  - $\bf{X}$ is original dataset, $\bf{P}$ is a transformation of $\bf{X}$ into $\bf{Y}$
- How to choose $\bf{P}$?<br>
  1) Reduce noise<br>
  2) Maximize variance

----

## Principle Component Analysis
# Concepts
<space>

- Covariance matrix<br>
$\bf{C} = \bf{XX}^{T}$

- Restated goals are
  - Minimize covariance and maximize variance
  - Optimal $\bf{C}$ is a diagonal matrix, off diagonals are = 0

----

## Principle Component Analysis
# Concepts
<space>

- Assumes linear relationship between $\bf{X}$ and $\bf{Y}$ (non-linear is a kernel PCA)
- Largest variance indicates most signal
- Orthogonal components - makes the linear algebra easier
- Assumes data is normally distributed, otherwise PCA might not diagonalize matrix
  - Can use ICA...
  - But most data is normal and PCA is robust to slight deviance from normality

----

## Principle Component Analysis
# Eigenwhat?
<space>

$Ax = \lambda x$
  - $\lambda$ is an eigenvalue of $A$ and $x$ is an eigenvector of $A$<br>
$Ax - \lambda Ix = 0$<br>
$(A - \lambda I)x = 0$<br>
$\det(A - \lambda I)$ = 0 <- roots of this yield eigenvalues of A

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X = \lambda X\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X = \lambda X\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X - \lambda X = 0\]
\[(\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix} - \lambda I)X = 0\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X = \lambda X\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X - \lambda X = 0\]
\[(\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix} - \lambda I)X = 0\]
\[\left | \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix} - \lambda I \right |= 0\]
\[\left|\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix} \right| = 0\]
\[\left|\begin{bmatrix} 5-\lambda & 2\\ 2 & 5-\lambda \end{bmatrix}\right| = 0\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

$(5-\lambda)\times(5-\lambda) - 4 = 0$
<br>
$\lambda^{2} - 10\lambda + 21 = 0$
<br>
$\lambda = ?$

----

## Principle Component Analysis
# Eigenwhat?
<space>


```r
A = matrix(c(5,2,2,5),nrow=2)
roots <- Re(polyroot(c(21,-10,1)))
roots
```

```
## [1] 3 7
```

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>
$x_{1} = -x_{2}$<br>

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>
$x_{1} = -x_{2}$<br>

\[Eigenvector = \begin{bmatrix} 1\\ -1 \end{bmatrix}\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when $\lambda = 7$<br>
$Ax = 7x$<br>
$5x_{1} + 2x_{2} = 7x_{1}$<br>
$2x_{2} + 5x_{2} = 7x_{2}$<br>
$x_{1} = x_{2}$<br>

\[Eigenvector = \begin{bmatrix} 1\\ 1 \end{bmatrix}\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

$Ax = \lambda x$

```r
A %*% c(1,-1) == 3 * as.matrix(c(1,-1))
```

```
##      [,1]
## [1,] TRUE
## [2,] TRUE
```

```r
A %*% c(1,1) == 7 * as.matrix(c(1,1))
```

```
##      [,1]
## [1,] TRUE
## [2,] TRUE
```

----

## Principle Component Analysis
# Eigenwhat?
<space>


```r
m <- matrix(c(1,-1,1,1),ncol=2)   ## two eigenvectors
m <- m/sqrt(norm(m))  ## normalize
as.matrix(m%*%diag(roots)%*%t(m))
```

```
##      [,1] [,2]
## [1,]    5    2
## [2,]    2    5
```

----

## Principle Component Analysis
# Motivation
<space>

$\bf{PX} = \bf{Y}$<br>
$\bf{C_{Y}} = \frac{1}{(n-1)}\bf{YY^{T}}$<br>

----

## Principle Component Analysis
# Motivation
<space>

$\bf{PX} = \bf{Y}$<br>
$\bf{C_{Y}} = \frac{1}{(n-1)}\bf{YY^{T}}$<br>
$=\bf{PX(PX)^{T}}$<br>
$=\bf{PXX^{T}P^{T}}$<br>

----

## Principle Component Analysis
# Motivation
<space>

$\bf{PX} = \bf{Y}$<br>
$\bf{C_{Y}} = \frac{1}{(n-1)}\bf{YY^{T}}$<br>
$=\bf{PX(PX)^{T}}$<br>
$=\bf{PXX^{T}P^{T}}$<br>
$=\bf{PAP^{T}}$<br>
- $\bf{P}$ is a matrix with columns that are eigenvectors
- $\bf{A}$ is a diagonalized matrix of eigenvalues and symmetric<br>
$\bf{A} = \bf{EDE^{T}}$

----

## Principle Component Analysis
# Motivation
<space>

- Each row of $\bf{P}$ should be an eigenvector of $\bf{A}$<br>
$\bf{P} = \bf{E^{T}}$
- Note that $\bf{P^{T}} = \bf{P^{-1}}$ (linear algebra)<br>
$\bf{A} = \bf{P^{T}DP}$<br>
$\bf{C_{Y}} = \bf{PP^{T}DPP^{T}} = \frac{1}{n-1}\bf{D}$
- $\bf{D}$ is a diagonal matrix, depending on how we choose $\bf{P}$
- Therefore $\bf{C_{Y}}$ is diagonalized

----

## Principle Component Analysis
# Example
<space>


```r
data <- read.csv('tennis_data_2013.csv')
```

```
## Warning: cannot open file 'tennis_data_2013.csv': No such file or
## directory
```

```
## Error: cannot open the connection
```

```r
data$Player1 <- as.character(data$Player1)
```

```
## Error: replacement has 0 rows, data has 6497
```

```r
data$Player2 <- as.character(data$Player2)
```

```
## Error: replacement has 0 rows, data has 6497
```

```r
tennis <- data
m <- length(data)

for (i in 10:m){
  tennis[,i] <- ifelse(is.na(data[,i]),0,data[,i])
}

str(tennis)
```

```
## 'data.frame':	6497 obs. of  13 variables:
##  $ type                : Factor w/ 2 levels "red","white": 1 1 1 1 1 1 1 1 1 1 ...
##  $ fixed.acidity       : num  7.4 7.8 7.8 11.2 7.4 7.4 7.9 7.3 7.8 7.5 ...
##  $ volatile.acidity    : num  0.7 0.88 0.76 0.28 0.7 0.66 0.6 0.65 0.58 0.5 ...
##  $ citric.acid         : num  0 0 0.04 0.56 0 0 0.06 0 0.02 0.36 ...
##  $ residual.sugar      : num  1.9 2.6 2.3 1.9 1.9 1.8 1.6 1.2 2 6.1 ...
##  $ chlorides           : num  0.076 0.098 0.092 0.075 0.076 0.075 0.069 0.065 0.073 0.071 ...
##  $ free.sulfur.dioxide : num  11 25 15 17 11 13 15 15 9 17 ...
##  $ total.sulfur.dioxide: num  34 67 54 60 34 40 59 21 18 102 ...
##  $ density             : num  0.998 0.997 0.997 0.998 0.998 ...
##  $ pH                  : num  3.51 3.2 3.26 3.16 3.51 3.51 3.3 3.39 3.36 3.35 ...
##  $ sulphates           : num  0.56 0.68 0.65 0.58 0.56 0.56 0.46 0.47 0.57 0.8 ...
##  $ alcohol             : num  9.4 9.8 9.8 9.8 9.4 9.4 9.4 10 9.5 10.5 ...
##  $ quality             : int  5 5 5 6 5 5 5 7 7 5 ...
```

```r
features <- tennis[,10:m]

head(features)
```

```
##     pH sulphates alcohol quality
## 1 3.51      0.56     9.4       5
## 2 3.20      0.68     9.8       5
## 3 3.26      0.65     9.8       5
## 4 3.16      0.58     9.8       6
## 5 3.51      0.56     9.4       5
## 6 3.51      0.56     9.4       5
```

```r
str(features)
```

```
## 'data.frame':	6497 obs. of  4 variables:
##  $ pH       : num  3.51 3.2 3.26 3.16 3.51 3.51 3.3 3.39 3.36 3.35 ...
##  $ sulphates: num  0.56 0.68 0.65 0.58 0.56 0.56 0.46 0.47 0.57 0.8 ...
##  $ alcohol  : num  9.4 9.8 9.8 9.8 9.4 9.4 9.4 10 9.5 10.5 ...
##  $ quality  : int  5 5 5 6 5 5 5 7 7 5 ...
```

```r
dim(features)
```

```
## [1] 6497    4
```

----

## Principle Component Analysis
# Example
<space>


```r
scaled_features <- as.matrix(scale(features))
Cx <- cov(scaled_features)
eigenvalues <- eigen(Cx)$values
eigenvectors <- eigen(Cx)$vectors
PC <- scaled_features %*% eigenvectors
```

----

## Principle Component Analysis
# Example
<space>


```r
Cy <- cov(PC)
sum(round(diag(Cy) - eigenvalues,5))
```

```
## [1] 0
```

```r
sum(round(Cy[upper.tri(Cy)],5)) ## off diagonals are 0 since PC's are orthogonal
```

```
## [1] 0
```

----

## Principle Component Analysis
# Example
<space>


```r
var_explained <- round(eigenvalues/sum(eigenvalues) * 100, digits = 2)
cum_var_explained <- round(cumsum(eigenvalues)/sum(eigenvalues) * 100, digits = 2)

var_explained <- as.data.frame(var_explained)
names(var_explained) <- "variance_explained"
var_explained$PC <- as.numeric(rownames(var_explained))
var_explained <- cbind(var_explained,cum_var_explained)

library(ggplot2)
ggplot(var_explained) +
  geom_bar(aes(x=PC,y=variance_explained),stat='identity') +
  geom_line(aes(x=PC,y=cum_var_explained))
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 

----

## Principle Component Analysis
# Example
<space>


```r
pca.df <- prcomp(scaled_features)
eigenvalues == (pca.df$sdev)^2
```

```
## [1] FALSE FALSE FALSE FALSE
```

```r
eigenvectors[,1] == pca.df$rotation[,1]
```

```
##        pH sulphates   alcohol   quality 
##     FALSE     FALSE     FALSE     FALSE
```

```r
sum((eigenvectors[,1])^2)
```

```
## [1] 1
```

----

## Principle Component Analysis
# Example
<space>


```r
rows <- nrow(tennis)
pca.plot <- as.data.frame(pca.df$x[,1:2])
pca.plot$gender <- data$Gender
ggplot(data=pca.plot,aes(x=PC1,y=PC2,color=gender)) + geom_point()
```

```
## Error: object 'gender' not found
```

----

## Principle Component Analysis
# Example
<space>

- how accurate is the first PC at dividing the dataset?
gen <- ifelse(pca.df$x[,1] > abs(mean(pca.df$x[,1]))*2,"F","M")
sum(diag(table(gen,as.character(data$Gender))))/rows

----

## Principle Component Analysis
# Summary
<space>

----

## Clustering
# Motivation
<space>

- separate data into meaningful or useful groups
  - capture natural structure of the data
  - starting point for further analysis
- cluster for utility
  - summarizing data for less expensive computation
  - data compression

----

## Clustering
# Types of Clusters
<space>

- data that looks more like other data in that cluster than outside
- each data point is more similar to the prototype (centeroid) of the cluster than the prototype of other clusters
- where the density is highest, that is a cluster

----

## Clustering
# Typical clustering problem
<space>

![plot of chunk cluster_plot_example](figure/cluster_plot_example.png) 

----

## Clustering
# Density based cluster
<space>

<img src="http://upload.wikimedia.org/wikipedia/commons/0/05/DBSCAN-density-data.svg" density_based />

----

## Clustering
# Kmeans algorithm
<space>

- Select K points as initial centroids 
- Do
  - Form K clusters by assigning each point to its closest centroid
  - Recompute the centroid of each cluster 
- Until centroids do not change, or change very minimally, i.e. <1%

----

## Clustering
# Kmeans algorithm
<space>

- Use similarity measures (Euclidean or cosine) depending on the data
- Minimize the squared distance of each point to closest centroid
$SSE(k) = \sum_{i=1}^{m}\sum_{j=1}^{n} (x_{ij} - \bar{x}_{kj})$

----

## Clustering
# Kmeans - notes
<space>

- Choose initial K randomly 
  - can lead to poor centroids - local minimuum
  - Run kmeans multiple times
- Reduce the total SSE by increasing K
- Increase the cluster with largest SSE
- Decrease K and minimize SSE
- Split up a cluster into other clusters
  - The centroid that is split will increase total SSE the least

----

## Clustering
# Kmeans
<space>

- Bisecting K means
  - Split points into 2 clusters
    - Take cluster with largest SSE - split that into two clusters
  - Rerun bisecting K mean on resulting clusters
  - Stop when you have K clusters
- Less susceptible to initialization problems

----

## Clustering
# Kmean fails
<space>

![different_density](C:/Users/Ilan%20Man/Desktop/Personal/RPres_ML_2/figure/different_density.png)

----

## Clustering
# Kmean fails
<space>

![different_size_clusters](C:/Users/Ilan%20Man/Desktop/Personal/RPres_ML_2/figure/different_size_clusters.png)

----

## Clustering
# Kmean fails
<space>

![non-globular](C:/Users/Ilan%20Man/Desktop/Personal/RPres_ML_2/figure/non-globular.png)

----

## Clustering
# Kmeans
<space>


```r
wine <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
```

```
## Warning: cannot open: HTTP status was '403 Forbidden'
```

```
## Error: cannot open the connection
```

```r
names(wine) <- c("class",'Alcohol','Malic','Ash','Alcalinity','Magnesium','Total_phenols',
                 'Flavanoids','NFphenols','Proanthocyanins','Color','Hue','Diluted','Proline')
str(wine)
```

```
## 'data.frame':	177 obs. of  14 variables:
##  $ class          : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ Alcohol        : num  13.2 13.2 14.4 13.2 14.2 ...
##  $ Malic          : num  1.78 2.36 1.95 2.59 1.76 1.87 2.15 1.64 1.35 2.16 ...
##  $ Ash            : num  2.14 2.67 2.5 2.87 2.45 2.45 2.61 2.17 2.27 2.3 ...
##  $ Alcalinity     : num  11.2 18.6 16.8 21 15.2 14.6 17.6 14 16 18 ...
##  $ Magnesium      : int  100 101 113 118 112 96 121 97 98 105 ...
##  $ Total_phenols  : num  2.65 2.8 3.85 2.8 3.27 2.5 2.6 2.8 2.98 2.95 ...
##  $ Flavanoids     : num  2.76 3.24 3.49 2.69 3.39 2.52 2.51 2.98 3.15 3.32 ...
##  $ NFphenols      : num  0.26 0.3 0.24 0.39 0.34 0.3 0.31 0.29 0.22 0.22 ...
##  $ Proanthocyanins: num  1.28 2.81 2.18 1.82 1.97 1.98 1.25 1.98 1.85 2.38 ...
##  $ Color          : num  4.38 5.68 7.8 4.32 6.75 5.25 5.05 5.2 7.22 5.75 ...
##  $ Hue            : num  1.05 1.03 0.86 1.04 1.05 1.02 1.06 1.08 1.01 1.25 ...
##  $ Diluted        : num  3.4 3.17 3.45 2.93 2.85 3.58 3.58 2.85 3.55 3.17 ...
##  $ Proline        : int  1050 1185 1480 735 1450 1290 1295 1045 1045 1510 ...
```

- set.seed() to make sure results are reproducible
- add nstart to the function call so that it attempts multiple configurations, selecting the best
- use a screeplot to select optimal K

----

## Clustering
# Kmeans
<space>


```r
s.wine <- scale(wine[,-1])
best_k <- 0
num_k <- 20
for (i in 1:num_k){
  best_k[i] <- sum(kmeans(s.wine,centers=i)$withinss)
  }

barplot(best_k, xlab = "Number of clusters",
        names.arg = 1:num_k,
        ylab="Within groups sum of squares",
        main="Scree Plot for Wine dataset")
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 

----

## Clustering
# Kmeans animation
<space>

install.packages('animation')
library(animation)

oopt = ani.options(interval = 1)
ani_ex = rbind(matrix(rnorm(100, sd = 0.3), ncol = 2), 
          matrix(rnorm(100, sd = 0.3), 
          ncol = 2))
colnames(ani_ex) = c("x", "y")

kmeans.an = function(
  x = cbind(X1 = runif(50), X2 = runif(50)), centers = 4, hints = c('Move centers!', 'Find cluster?'),
  pch = 1:5, col = 1:5
) {
  x = as.matrix(x)
  ocluster = sample(centers, nrow(x), replace = TRUE)
  if (length(centers) == 1) centers = x[sample(nrow(x), centers), ] else
    centers = as.matrix(centers)
  numcent = nrow(centers)
  dst = matrix(nrow = nrow(x), ncol = numcent)
  j = 1
  pch = rep(pch, length = numcent)
  col = rep(col, length = numcent)
  
  for (j in 1:ani.options('nmax')) {
    dev.hold()
    plot(x, pch = pch[ocluster], col = col[ocluster], panel.first = grid())
    mtext(hints[1], 4)
    points(centers, pch = pch[1:numcent], cex = 3, lwd = 2, col = col[1:numcent])
    ani.pause()
    for (i in 1:numcent) {
      dst[, i] = sqrt(apply((t(t(x) - unlist(centers[i, ])))^2, 1, sum))
    }
    ncluster = apply(dst, 1, which.min)
    plot(x, type = 'n')
    mtext(hints[2], 4)
    grid()
    ocenters = centers
    for (i in 1:numcent) {
      xx = subset(x, ncluster == i)
      polygon(xx[chull(xx), ], density = 10, col = col[i], lty = 2)
      points(xx, pch = pch[i], col = col[i])
      centers[i, ] = apply(xx, 2, mean)
    }
    points(ocenters, cex = 3, col = col[1:numcent], pch = pch[1:numcent], lwd = 2)
    ani.pause()
    if (all(ncluster == ocluster)) break
    ocluster = ncluster
  }
  invisible(list(cluster = ncluster, centers = centers))
}

kmeans.an(ani_ex, centers = 5, hints = c("Move centers","Cluster found?"))

----

## Clustering
# K-medoid
<space>

- multiple distance metrics
- robust medioids
- computationally expensive
- cluster center is one of the points itself

----

## Clustering
# K-medoid
<space>

- cluster each point based on the closest center
- replace each center by the medioid of points in its cluster

----

## Clustering
# K-medoid
<space>

- Selecting the optimal number of clusters
- For each point p, first find the average distance between p and all other points in the same cluster, $A$
- Then find the average distance between p and all points in the nearest cluster, $B$
- The silhouette coefficient for p is $\frac{A - B}{\max(A,B)}$
  - Values close to 1 mean point clearly belongs to that cluster
  - Values close to 0 mean points might belong in another cluster

----

## Clustering
# K-medoid
<space>


```r
library(cluster)

pam.best <- as.numeric()
for (i in 2:20){
  pam.best[i] <- pam(s.wine, k=i)$silinfo$avg.width
}
best_k <- which.max(pam.best)
best_k
```

```
## [1] 3
```

----

## Clustering
# K-medoid
<space>


```r
clusplot(pam(s.wine,best_k), main="K-medoids with K = 3",sub=NULL)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7.png) 

----

## Clustering
# DBSCAN
<space>

- A cluster is a dense region of points separated by low-density regions
- Group objects into one cluster if they are connected to one another by densely populated area
- Used when the clusters are irregular or intertwined, and when noise and outliers are present

----

## Clustering
# Terminology
<space>

- Core points are located inside a cluster
- Border points are on the borders between two clusters
- Neighborhood of p are all points within some radius of p, Eps

----

## Clustering
# Terminology
<space>

- Core points are located inside a cluster
- Border points are on the borders between two clusters
- Neighborhood of p are all points within some radius of p, Eps
![density](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/density_structure.png)

----

## Clustering
# Terminology
<space>

- Core points are located inside a cluster
- Border points are on the borders between two clusters
- Neighborhood of p are all points within some radius of p, Eps
- High density region has at least Minpts within Eps of point p
- Noise points are not within Eps of border or core points

----

## Clustering
# Terminology
<space>

- Core points are located inside a cluster
- Border points are on the borders between two clusters
- Neighborhood of p are all points within some radius of p, Eps
- High density region has at least Minpts within Eps of point p
- Noise points are not within Eps of border or core points
- If p is density connected to q, they are part of the same cluster, if not, then they are not
- If p is not density connected to any other point, its considered noise

----

## Clustering
# DBSCAN
<space>

![density_win](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/density_ex_win.png)

----

## Clustering
# DBSCAN
<space>


```r
x <- c(2,2,8,5,7,6,1,4)
y <- c(10,5,4,8,5,4,2,9)
cluster <- data.frame(X=c(x,2*x,3*x),Y=c(y,-2*x,1/4*y))
plot(cluster)
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8.png) 

----

## Clustering
# DBSCAN
<space>


```r
library(fpc)
cluster_DBSCAN<-dbscan(cluster, eps=3, MinPts=2, method="hybrid")
plot(cluster_DBSCAN, cluster, main="Clustering using DBSCAN algorithm (eps=3, MinPts=3)")
```

![plot of chunk dbscan_ex](figure/dbscan_ex.png) 

----

## Clustering
# Summary
<space>

- Unsupervised learning
- Not a perfect science - lots of interpretation
- Hard to define "correct" clustering
- Many types of algorithms

----

## Trees
# Motivation
<space>

- representation of decisions made in order to classify or predict
![overview](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/tree_example.png)

----

## Trees
# Structure
<space>

![structure](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/tree_structure.png)

----

## Trees
# Structure
<space>

- recursive partitioning -> "divide and conquer"
- going down, choose feature that is most *predictive* of target class
  - split the data according to feature
  - continue...

----

## Trees
# Structure
<space>

until...
- all examples at a node are in same class
- no more features left to distinguish (prone to overfitting)
- tree has grown to some prespecified limit (prune)

----

## Trees
# Algorithms
<space>

- ID3
  - original, popular, DT implementation
- C4.5
  - like ID3 +
  - handles continuous cases
  - imputing missing values
  - weighing costs
  - pruning post creation
- C5.0
  - like C4.5 + 
  - faster, less memory usage
  - boosting

----

## Trees
# Selecting features
<space>

- How does tree decide how to select feature?
  - purity of resulting split
- __Entropy__: amount of information contained in a random variable
  - For a feature with N classes:
    - 0 = purely homogenous
    - $\log_{2}(N)$ = completely mixed

----

## Trees
# Entropy
<space>

$Entropy(S) = \sum_{i=1}^{c} -p_{i}\log_{2}(p_{i})$
  - where $S$ is a dataset
  - $c$ is the number of levels in that data
  - $p_{i}$ is the proportion of values in that level

----

## Trees
# Entropy - example
<space>

What is the entropy of a fair, 6 sided die?


```r
entropy <- function(probs){
  ent <- 0
  for(i in probs){
    ent_temp <- -i*log2(i)
    ent <- ent + ent_temp
  }
  return(ent)
}
```

----

## Trees
# Entropy - example
<space>


```r
fair <- rep(1/6,6)
entropy(fair)
```

```
## [1] 2.585
```

```r
log2(6)
```

```
## [1] 2.585
```

----

## Trees
# Entropy - example
<space>

What is the entropy of a biased, 6 sided die?
- $P(X=1) = P(X=2) = P(X=3) = 1/9$
- $P(X=4) = P(X=5) = P(X=6) = 2/9$


```r
biased <- c(rep(1/9,3),rep(2/9,3))
entropy(biased)
```

```
[1] 2.503
```

----

## Trees
# Entropy - example
<space>


```r
more_biased <- c(rep(1/18,3),rep(5/18,3))
entropy(more_biased)
```

```
[1] 2.235
```

```r
most_biased <- c(rep(1/100,5),rep(95/100,1))
entropy(most_biased)
```

```
[1] 0.4025
```

----

## Trees
# Entropy - example
<space>


```r
curve(-x*log2(x)-(1 - x)*log2(1 - x), col =" red", xlab = "x", ylab = "Entropy", 
      lwd = 4, main='Entropy of a coin toss')
```

![plot of chunk entropy_curve](figure/entropy_curve.png) 

----

## Trees
# Entropy
<space>

- C5.0 uses the change in entropy to determine the change in purity
- InfoGain = Entropy (pre split) - Entropy (post split)
  - Entropy (pre split) = current Entropy
  - Entropy (post split) is trickier
    - need to consider Entropy of each possible split
  - $E(post) = \sum_{i=1}^{n}w_{i}Entropy(P_{i})$

- Notes:
  - The more a feature splits the data in obvious ways, the less informative it is, entropy is lower
  - The more a feature splits the data - in general - the higher the entropy and hence information gained by splitting at that feature

----

## Trees
# Example
<space>


```r
voting_data <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')
```

```
## Warning: cannot open: HTTP status was '403 Forbidden'
```

```
## Error: cannot open the connection
```

```r
names(voting_data) <- c('party','handicapped-infants','water-project-cost-sharing',
                        'adoption-of-the-budget-resolution','physician-fee-freeze',
                        'el-salvador-aid','religious-groups-in-schools',
                        'anti-satellite-test-ban','aid-to-nicaraguan-contras',
                        'mx-missile','immigration','synfuels-corporation-cutback',
                        'education-spending','superfund-right-to-sue','crime',
                        'duty-free-exports','export-administration-act-south-africa')
```

----

## Trees
# Example
<space>


```r
prop.table(table(voting_data[,1]))
```

```

  democrat republican 
    0.6152     0.3848 
```

```r
n <- nrow(voting_data)
train_ind <- sample(n,2/3*n)
voting_train <- voting_data[train_ind,]
voting_test <- voting_data[-train_ind,]
```

----

## Trees
# Example
<space>

<img src="/Users/ilanman/Desktop/Data/RPres_ML_2/figure/real_tree_example.png" height="500px" width="500px" />


----

## Trees
# Example
<space>


```

 
   Cell Contents
|-------------------------|
|                       N |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  145 

 
             | predicted class 
actual class |   democrat | republican |  Row Total | 
-------------|------------|------------|------------|
    democrat |         77 |          0 |         77 | 
             |      0.531 |      0.000 |            | 
-------------|------------|------------|------------|
  republican |         12 |         56 |         68 | 
             |      0.083 |      0.386 |            | 
-------------|------------|------------|------------|
Column Total |         89 |         56 |        145 | 
-------------|------------|------------|------------|

 
```

----

## Trees
# Example
<space>


```r
# most important variables
head(C5imp(tree_model))
```

```
##                                   Overall
## physician-fee-freeze                98.27
## el-salvador-aid                     39.45
## anti-satellite-test-ban             37.02
## adoption-of-the-budget-resolution   29.07
## handicapped-infants                  0.00
## water-project-cost-sharing           0.00
```

----

## Trees
# Example
<space>


```r
# in-sample error rate
summary(tree_model)
```

```
## 
## Call:
## C5.0.default(x = voting_train[, -1], y = voting_train[, 1], trials = 1)
## 
## 
## C5.0 [Release 2.07 GPL Edition]  	Wed Aug 20 20:54:00 2014
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 289 cases (17 attributes) from undefined.data
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (176)
## physician-fee-freeze = y:
## :...el-salvador-aid = ?: republican (0)
##     el-salvador-aid = n: democrat (4.2/1)
##     el-salvador-aid = y:
##     :...anti-satellite-test-ban in {?,y}: republican (24.7)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (76.5/5.3)
##             adoption-of-the-budget-resolution = y: democrat (7.5/2)
## 
## 
## Evaluation on training data (289 cases):
## 
## 	    Decision Tree   
## 	  ----------------  
## 	  Size      Errors  
## 
## 	     5    8( 2.8%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	   185     5    (a): class democrat
## 	     3    96    (b): class republican
## 
## 
## 	Attribute usage:
## 
## 	 98.27%	physician-fee-freeze
## 	 39.45%	el-salvador-aid
## 	 37.02%	anti-satellite-test-ban
## 	 29.07%	adoption-of-the-budget-resolution
## 
## 
## Time: 0.0 secs
```

----

## Trees
# Boosting
<space>

- by combining a number of weak performing learners create a team that is much stronger than any one of the learners alone.
- this is where C5.0 improves on C4.5

----

## Trees
# Example - Boosting
<space>


```r
boosted_tree_model <- C5.0(voting_train[,-1],voting_train[,1], trials=25)
boosted_tennis_predict <- predict(boosted_tree_model,voting_test[,-1])

boosted_conf <- CrossTable(voting_test[,1], boosted_tennis_predict, prop.chisq = FALSE,
                           prop.c = FALSE, prop.r = FALSE, 
                           dnn = c("actual class", "predicted class"))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  145 
## 
##  
##              | predicted class 
## actual class |   democrat | republican |  Row Total | 
## -------------|------------|------------|------------|
##     democrat |         76 |          1 |         77 | 
##              |      0.524 |      0.007 |            | 
## -------------|------------|------------|------------|
##   republican |          5 |         63 |         68 | 
##              |      0.034 |      0.434 |            | 
## -------------|------------|------------|------------|
## Column Total |         81 |         64 |        145 | 
## -------------|------------|------------|------------|
## 
## 
```

----

## Trees
# Example - Boosting
<space>


```r
# in-sample error rate
summary(boosted_tree_model)
```

```
## 
## Call:
## C5.0.default(x = voting_train[, -1], y = voting_train[, 1], trials = 25)
## 
## 
## C5.0 [Release 2.07 GPL Edition]  	Wed Aug 20 20:54:00 2014
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 289 cases (17 attributes) from undefined.data
## 
## -----  Trial 0:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (176)
## physician-fee-freeze = y:
## :...el-salvador-aid = ?: republican (0)
##     el-salvador-aid = n: democrat (4.2/1)
##     el-salvador-aid = y:
##     :...anti-satellite-test-ban in {?,y}: republican (24.7)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (76.5/5.3)
##             adoption-of-the-budget-resolution = y: democrat (7.5/2)
## 
## -----  Trial 1:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (132.7)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (70.3/2.3)
##     immigration = n:
##     :...synfuels-corporation-cutback = n: republican (38.6/11)
##         synfuels-corporation-cutback in {?,y}: democrat (47.3/5.7)
## 
## -----  Trial 2:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (104.9)
## physician-fee-freeze = y:
## :...duty-free-exports in {?,n}: republican (146.4/44.4)
##     duty-free-exports = y: democrat (37.7/10.5)
## 
## -----  Trial 3:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (83.4)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (77.8/17.4)
##     immigration = n:
##     :...mx-missile in {?,y}: democrat (19.5/0.9)
##         mx-missile = n:
##         :...anti-satellite-test-ban in {?,y}: republican (14.7)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution in {?,y}: democrat (11.8)
##                 adoption-of-the-budget-resolution = n:
##                 :...superfund-right-to-sue in {?,n}: democrat (12.5/0.5)
##                     superfund-right-to-sue = y: republican (69.2/30.2)
## 
## -----  Trial 4:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (68.5)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback = ?: republican (0)
##     synfuels-corporation-cutback = n:
##     :...education-spending = n: democrat (23/8.1)
##     :   education-spending in {?,y}: republican (76.8/6.2)
##     synfuels-corporation-cutback = y:
##     :...religious-groups-in-schools = ?: democrat (0)
##         religious-groups-in-schools = n: republican (7.5/2.8)
##         religious-groups-in-schools = y:
##         :...mx-missile in {?,y}: democrat (32.5)
##             mx-missile = n:
##             :...anti-satellite-test-ban in {?,n}: democrat (70.6/19.1)
##                 anti-satellite-test-ban = y: republican (10.1)
## 
## -----  Trial 5:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (52.8)
## physician-fee-freeze = y:
## :...el-salvador-aid = ?: republican (0)
##     el-salvador-aid = n: democrat (21.5/3.7)
##     el-salvador-aid = y:
##     :...anti-satellite-test-ban in {?,y}: republican (48.2)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution = ?: republican (0)
##             adoption-of-the-budget-resolution = y: democrat (40.2/7.3)
##             adoption-of-the-budget-resolution = n:
##             :...immigration in {?,y}: republican (30.5)
##                 immigration = n:
##                 :...mx-missile = ?: republican (0)
##                     mx-missile = y: democrat (4.9)
##                     mx-missile = n:
##                     :...religious-groups-in-schools in {?,
##                         :                               n}: republican (4.4)
##                         religious-groups-in-schools = y:
##                         :...superfund-right-to-sue = ?: republican (0)
##                             superfund-right-to-sue = n: democrat (7.3/0.4)
##                             superfund-right-to-sue = y:
##                             :...education-spending = n: democrat (28.6/10.2)
##                                 education-spending in {?,
##                                                        y}: republican (50.6/12.1)
## 
## -----  Trial 6:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (40.8)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (112.2/16.5)
##     synfuels-corporation-cutback = y:
##     :...export-administration-act-south-africa in {?,n}: democrat (54.9/16.1)
##         export-administration-act-south-africa = y:
##         :...mx-missile in {?,n}: republican (49/10.8)
##             mx-missile = y: democrat (32.1/13)
## 
## -----  Trial 7:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (34.2)
## physician-fee-freeze = y:
## :...religious-groups-in-schools in {?,n}: republican (29.4/3.7)
##     religious-groups-in-schools = y:
##     :...el-salvador-aid = ?: republican (0)
##         el-salvador-aid = n: democrat (12.1)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban in {?,y}: republican (23.8)
##             anti-satellite-test-ban = n:
##             :...duty-free-exports = ?: republican (0)
##                 duty-free-exports = y: democrat (25.1/3.1)
##                 duty-free-exports = n:
##                 :...immigration in {?,y}: republican (53.1/6.5)
##                     immigration = n: [S1]
## 
## SubTree [S1]
## 
## adoption-of-the-budget-resolution = n: republican (95.1/40.1)
## adoption-of-the-budget-resolution in {?,y}: democrat (16.2)
## 
## -----  Trial 8:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (27)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (98.4/21.2)
##     synfuels-corporation-cutback = y:
##     :...religious-groups-in-schools = ?: democrat (0)
##         religious-groups-in-schools = n: republican (23.5/10.6)
##         religious-groups-in-schools = y:
##         :...mx-missile in {?,y}: democrat (24.2)
##             mx-missile = n:
##             :...adoption-of-the-budget-resolution in {?,
##                 :                                     y}: democrat (11.3/0.2)
##                 adoption-of-the-budget-resolution = n:
##                 :...immigration = ?: democrat (0)
##                     immigration = y: republican (14.2)
##                     immigration = n: [S1]
## 
## SubTree [S1]
## 
## export-administration-act-south-africa in {?,n}: democrat (61.9/13)
## export-administration-act-south-africa = y: republican (28.6/8.3)
## 
## -----  Trial 9:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (21.7)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (92.6/24)
##     immigration = n:
##     :...el-salvador-aid in {?,n}: democrat (17.1)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban = ?: democrat (0)
##             anti-satellite-test-ban = y: republican (8.4)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution in {?,
##                 :                                     y}: democrat (28.2)
##                 adoption-of-the-budget-resolution = n:
##                 :...superfund-right-to-sue in {?,n}: democrat (12/2)
##                     superfund-right-to-sue = y:
##                     :...education-spending = n: democrat (48.1/15.1)
##                         education-spending in {?,y}: republican (60.8/20.7)
## 
## -----  Trial 10:  -----
## 
## Decision tree:
## 
## mx-missile in {?,y}: democrat (88.8/18)
## mx-missile = n:
## :...anti-satellite-test-ban in {?,y}: republican (22.8/1.9)
##     anti-satellite-test-ban = n:
##     :...religious-groups-in-schools in {?,n}: republican (7)
##         religious-groups-in-schools = y:
##         :...adoption-of-the-budget-resolution in {?,y}: democrat (32.6/5.5)
##             adoption-of-the-budget-resolution = n:
##             :...immigration in {?,y}: republican (18.7/0.3)
##                 immigration = n:
##                 :...crime in {?,n}: democrat (5.5/0.1)
##                     crime = y:
##                     :...handicapped-infants = n: democrat (63.7/21.5)
##                         handicapped-infants in {?,y}: republican (49.8/13)
## 
## -----  Trial 11:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (34.1)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (105.5/26.8)
##     synfuels-corporation-cutback = y:
##     :...duty-free-exports in {?,y}: democrat (16.1/1.4)
##         duty-free-exports = n:
##         :...mx-missile in {?,y}: democrat (39.1/8.9)
##             mx-missile = n:
##             :...immigration = n: democrat (80.7/34.5)
##                 immigration in {?,y}: republican (13.5)
## 
## -----  Trial 12:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (27.8)
## physician-fee-freeze = y:
## :...mx-missile = ?: republican (0)
##     mx-missile = y: democrat (67.3/24.7)
##     mx-missile = n:
##     :...anti-satellite-test-ban in {?,y}: republican (21.6)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution = ?: republican (0)
##             adoption-of-the-budget-resolution = y: democrat (27.3/5.1)
##             adoption-of-the-budget-resolution = n:
##             :...immigration in {?,y}: republican (12.5)
##                 immigration = n:
##                 :...crime = ?: republican (0)
##                     crime = n: democrat (3.5/0.1)
##                     crime = y:
##                     :...duty-free-exports = ?: republican (0)
##                         duty-free-exports = y: democrat (17.4/5.7)
##                         duty-free-exports = n: [S1]
## 
## SubTree [S1]
## 
## synfuels-corporation-cutback in {?,n}: republican (27.9)
## synfuels-corporation-cutback = y:
## :...water-project-cost-sharing = n: democrat (22.8/6.1)
##     water-project-cost-sharing in {?,y}: republican (60.9/15.8)
## 
## -----  Trial 13:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (22.3)
## physician-fee-freeze = y:
## :...education-spending = ?: republican (0)
##     education-spending = n:
##     :...duty-free-exports = n: republican (69.3/30.4)
##     :   duty-free-exports in {?,y}: democrat (23.8/2.2)
##     education-spending = y:
##     :...anti-satellite-test-ban in {?,y}: republican (40)
##         anti-satellite-test-ban = n:
##         :...aid-to-nicaraguan-contras in {?,n}: republican (124.2/39.6)
##             aid-to-nicaraguan-contras = y: democrat (9.5/0.7)
## 
## -----  Trial 14:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (19.1)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (111.1/30.9)
##     synfuels-corporation-cutback = y:
##     :...duty-free-exports in {?,y}: democrat (15.5/3.3)
##         duty-free-exports = n:
##         :...immigration in {?,n}: democrat (111/40.8)
##             immigration = y: republican (32.4/12)
## 
## -----  Trial 15:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (16.4)
## physician-fee-freeze = y:
## :...education-spending = ?: republican (0)
##     education-spending = n:
##     :...immigration in {?,n}: democrat (68.6/17.7)
##     :   immigration = y: republican (34/13.4)
##     education-spending = y:
##     :...anti-satellite-test-ban in {?,y}: republican (34.2)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (87.5/24.4)
##             adoption-of-the-budget-resolution = y: democrat (48.2/11.4)
## 
## -----  Trial 16:  -----
## 
## Decision tree:
## 
## el-salvador-aid = ?: republican (0)
## el-salvador-aid = n: democrat (48.2/9.1)
## el-salvador-aid = y:
## :...anti-satellite-test-ban in {?,y}: republican (41.3/1.6)
##     anti-satellite-test-ban = n:
##     :...aid-to-nicaraguan-contras in {?,y}: democrat (8.5/0.4)
##         aid-to-nicaraguan-contras = n:
##         :...immigration in {?,y}: republican (35.2/7.1)
##             immigration = n:
##             :...adoption-of-the-budget-resolution in {?,
##                 :                                     y}: democrat (26.3)
##                 adoption-of-the-budget-resolution = n:
##                 :...superfund-right-to-sue = ?: republican (0)
##                     superfund-right-to-sue = n: democrat (17.6/2.4)
##                     superfund-right-to-sue = y:
##                     :...mx-missile in {?,n}: republican (109.6/41.2)
##                         mx-missile = y: democrat (2.3)
## 
## -----  Trial 17:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (49.3)
## physician-fee-freeze = y:
## :...religious-groups-in-schools in {?,n}: republican (25.9/5.7)
##     religious-groups-in-schools = y:
##     :...el-salvador-aid in {?,n}: democrat (15.8)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban in {?,y}: republican (26.2)
##             anti-satellite-test-ban = n:
##             :...superfund-right-to-sue in {?,n}: democrat (12/0.4)
##                 superfund-right-to-sue = y:
##                 :...duty-free-exports in {?,y}: democrat (28/5.3)
##                     duty-free-exports = n:
##                     :...aid-to-nicaraguan-contras = ?: republican (0)
##                         aid-to-nicaraguan-contras = y: democrat (6.9/0.4)
##                         aid-to-nicaraguan-contras = n:
##                         :...immigration in {?,y}: republican (21.1)
##                             immigration = n: [S1]
## 
## SubTree [S1]
## 
## adoption-of-the-budget-resolution = n: republican (89.1/37.2)
## adoption-of-the-budget-resolution in {?,y}: democrat (14.8)
## 
## -----  Trial 18:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (38.5)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (104.8/24.5)
##     synfuels-corporation-cutback = y:
##     :...export-administration-act-south-africa in {?,n}: democrat (62.7/13)
##         export-administration-act-south-africa = y:
##         :...water-project-cost-sharing = n: republican (13.4/1.1)
##             water-project-cost-sharing in {?,y}: democrat (69.7/28.8)
## 
## -----  Trial 19:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (31.1)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (92/19.9)
##     immigration = n:
##     :...el-salvador-aid in {?,n}: democrat (9.4)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban = ?: democrat (0)
##             anti-satellite-test-ban = y: republican (17.2)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution in {?,
##                 :                                     y}: democrat (22)
##                 adoption-of-the-budget-resolution = n:
##                 :...handicapped-infants = n: democrat (66.3/27.7)
##                     handicapped-infants in {?,y}: republican (51/18.2)
## 
## -----  Trial 20:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (25.7)
## physician-fee-freeze = y:
## :...education-spending = ?: republican (0)
##     education-spending = n: democrat (89.5/33.7)
##     education-spending = y:
##     :...adoption-of-the-budget-resolution in {?,n}: republican (138.6/24.2)
##         adoption-of-the-budget-resolution = y: democrat (35.2/13.1)
## 
## -----  Trial 21:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (21.2)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (125.2/23.7)
##     synfuels-corporation-cutback = y:
##     :...export-administration-act-south-africa = ?: republican (0)
##         export-administration-act-south-africa = n: democrat (63.4/22.6)
##         export-administration-act-south-africa = y:
##         :...mx-missile in {?,n}: republican (51.7/12.9)
##             mx-missile = y: democrat (27.6/10.9)
## 
## -----  Trial 22:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (18.2)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (98.6/23.9)
##     immigration = n:
##     :...mx-missile = ?: republican (0)
##         mx-missile = y: democrat (19.8/3.2)
##         mx-missile = n:
##         :...anti-satellite-test-ban in {?,y}: republican (12.8)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution = ?: republican (0)
##                 adoption-of-the-budget-resolution = y: democrat (14.5)
##                 adoption-of-the-budget-resolution = n:
##                 :...education-spending = n: democrat (45.7/16.5)
##                     education-spending in {?,y}: republican (79.3/22.1)
## 
## -----  Trial 23:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (15.4)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (112.1/28.2)
##     synfuels-corporation-cutback = y:
##     :...duty-free-exports in {?,y}: democrat (21.3/2.9)
##         duty-free-exports = n:
##         :...religious-groups-in-schools = ?: democrat (0)
##             religious-groups-in-schools = n: republican (15.4/4.4)
##             religious-groups-in-schools = y:
##             :...mx-missile in {?,y}: democrat (18.4)
##                 mx-missile = n:
##                 :...immigration in {?,y}: republican (17)
##                     immigration = n:
##                     :...superfund-right-to-sue in {?,n}: democrat (10.3)
##                         superfund-right-to-sue = y: [S1]
## 
## SubTree [S1]
## 
## water-project-cost-sharing = n: democrat (22.9/3.1)
## water-project-cost-sharing in {?,y}: republican (56.3/19.7)
## 
## -----  Trial 24:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (13.1)
## physician-fee-freeze = y:
## :...adoption-of-the-budget-resolution = ?: republican (0)
##     adoption-of-the-budget-resolution = y:
##     :...anti-satellite-test-ban in {?,n}: democrat (68.2/6.9)
##     :   anti-satellite-test-ban = y: republican (37/10.2)
##     adoption-of-the-budget-resolution = n:
##     :...el-salvador-aid = ?: republican (0)
##         el-salvador-aid = n: democrat (17.7)
##         el-salvador-aid = y:
##         :...immigration in {?,y}: republican (26.3)
##             immigration = n:
##             :...crime = ?: republican (0)
##                 crime = n: democrat (4.9/0.1)
##                 crime = y:
##                 :...handicapped-infants in {?,y}: republican (32.8/1.5)
##                     handicapped-infants = n:
##                     :...education-spending = n: democrat (14.4/0.3)
##                         education-spending in {?,y}: republican (73.5/24.8)
## 
## 
## Evaluation on training data (289 cases):
## 
## Trial	    Decision Tree   
## -----	  ----------------  
## 	  Size      Errors  
## 
##    0	     5    8( 2.8%)
##    1	     4   15( 5.2%)
##    2	     3   16( 5.5%)
##    3	     7   10( 3.5%)
##    4	     7   19( 6.6%)
##    5	    10    6( 2.1%)
##    6	     5   13( 4.5%)
##    7	     8    7( 2.4%)
##    8	     8    9( 3.1%)
##    9	     8    8( 2.8%)
##   10	     8   72(24.9%)
##   11	     6   11( 3.8%)
##   12	    10   15( 5.2%)
##   13	     6   13( 4.5%)
##   14	     5   12( 4.2%)
##   15	     6    9( 3.1%)
##   16	     8   31(10.7%)
##   17	    10    9( 3.1%)
##   18	     5   14( 4.8%)
##   19	     7   37(12.8%)
##   20	     4   23( 8.0%)
##   21	     5   13( 4.5%)
##   22	     7   10( 3.5%)
##   23	     9   10( 3.5%)
##   24	     9    8( 2.8%)
## boost	          2( 0.7%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	   188     2    (a): class democrat
## 	          99    (b): class republican
## 
## 
## 	Attribute usage:
## 
## 	 98.27%	physician-fee-freeze
## 	 96.54%	el-salvador-aid
## 	 95.16%	mx-missile
## 	 56.75%	anti-satellite-test-ban
## 	 48.10%	religious-groups-in-schools
## 	 47.06%	adoption-of-the-budget-resolution
## 	 46.71%	immigration
## 	 38.75%	synfuels-corporation-cutback
## 	 38.06%	education-spending
## 	 38.06%	duty-free-exports
## 	 36.33%	aid-to-nicaraguan-contras
## 	 30.80%	superfund-right-to-sue
## 	 17.99%	handicapped-infants
## 	 17.30%	crime
## 	  8.65%	water-project-cost-sharing
## 	  8.65%	export-administration-act-south-africa
## 
## 
## Time: 0.0 secs
```

----

## Trees
# Error Cost
<space>

- still getting too many false positives (predict republican but actually democrat)
- introduce higher cost to getting this wrong


```r
error_cost <- matrix(c(0,1,2,0),nrow=2)
cost_model <- C5.0(voting_train[,-1],voting_train[,1], trials=1, costs = error_cost)
```

```
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
```

```r
cost_predict <- predict(cost_model, newdata=voting_test[,-1])
conf <- CrossTable(voting_test[,1], cost_predict, prop.chisq = FALSE,
                   prop.c = FALSE, prop.r = FALSE,
                   dnn = c("actual class", "predicted class"))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  145 
## 
##  
##              | predicted class 
## actual class |   democrat | republican |  Row Total | 
## -------------|------------|------------|------------|
##     democrat |         72 |          5 |         77 | 
##              |      0.497 |      0.034 |            | 
## -------------|------------|------------|------------|
##   republican |          2 |         66 |         68 | 
##              |      0.014 |      0.455 |            | 
## -------------|------------|------------|------------|
## Column Total |         74 |         71 |        145 | 
## -------------|------------|------------|------------|
## 
## 
```

----

## Trees
# Error Cost
<space>


```
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
## 
## Warning: 
## no dimnames were given for the cost matrix; the factor levels will be used
```

![plot of chunk plot_boost_acc](figure/plot_boost_acc.png) 

----

## Trees
# Pros and Cons
<space>

- trees are non-parametric, rule based classification or regression method
- simple to understand and interpret
- little data preparation
- works well with small or large number of features
<br>
- easy to overfit
- biased towards splits on features with large number of levels
- usually finds local optimum
- difficult concepts are hard to learn
- avoid pre-pruning
- hard to know optimal length of tree without growing it there first

----

## Summary
# ML - Part II
<space>

- Logistic regression
- Math behind PCA
- 3 types of clusters
- Trees and improvements

----

## Resources
<space>

- [Machine Learning with R](http://www.packtpub.com/machine-learning-with-r/book)
- [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
- [Elements of Statistical Learning](http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)

----
