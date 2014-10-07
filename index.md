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
# Concepts
<space>

- Binary response variable (Y = 1 or Y = 0) association to a set of explanatory variables
- Like Linear Regression with a categorical outcome

----

## Logistic Regression
# Concepts
<space>

- Binary response variable (Y = 1 or Y = 0) association to a set of explanatory variables
- Like Linear Regression with a categorical outcome
- $\hat{y} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$

----

## Logistic Regression
# Concepts
<space>

- Binary response variable (Y = 1 or Y = 0) association to a set of explanatory variables
- Like Linear Regression with a categorical outcome
- $\hat{y} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$ becomes<br>
- $\log{\frac{P(Y=1)}{1 - P(Y=1)}} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$
- Can be extended to multiple and/or ordered categories

----

## Logistic Regression
# Concepts
<space>

- Family of GLMs
<ol>
<li>Random component <br>- Noise or Errors
<li>Systematic Component <br>- Linear combination in $X_{i}$
<li>Link Function <br>- Connects Random and Systematic components
</ol>

----

## Logistic Regression
# Concepts
<space>

- Data is I.I.D.
  - $Y$'s assume to come from family of exponential distributions
- Uses MLE to determine parameters - Not OLS
  - MLE satisfies lots of nice properties (unbiased, consistent)
  - Does not require transformation of $Y$'s to be Normal
  - Does not require constant variance

----

## Logistic Regression
# Concepts
<space>

- Type of regression to predict the probability of being in a class
  - Output is $P(Y=1 | X)$
  - Typical threshold is 0.5
- Assumes error terms are Binomially distributed
  - Generates 1's and 0's as the error term

----

## Logistic Regression
# Concepts
<space>

- Type of regression to predict the probability of being in a class
  - Output is $P(Y=1\hspace{2 mm} |\hspace{2 mm} X)$
  - Typical threshold is 0.5
- Assumes error terms are Binomially distributed
  - Generates 1's and 0's as the error term
- Sigmoid (logistic) function: $g(z) = \frac{1}{1+e^{-z}}$
  - Bounded by 0 and 1

----

## Logistic Regression
# Concepts
<space>

![plot of chunk log_curve](figure/log_curve.png) 

----

## Logistic Regression
# Find parameters
<space>

- The hypothesis function, $h_{\theta}(x)$, is $P(Y=1|X)$
- Linear regression: $h_{\theta}(x) = \theta x^{T}$<br>
(Recall that $\theta x^{T} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$)

----

## Logistic Regression
# Find parameters
<space>

- The hypothesis function, $h_{\theta}(x)$, is $P(Y=1|X)$
- Linear regression: $h_{\theta}(x) = \theta x^{T}$<br>
(Recall that $\theta x^{T} = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{n}x_{n}$)
- Logistic regression: $h_{\theta}(x) = g(\theta x^{T})$ 
<br>
where $g(z) = \frac{1}{1+e^{-z}}$

----

## Logistic Regression
# Notation
<space>

- Re-arranging $Y = \frac{1}{1+e^{-\theta x^{T}}}$ yields
<br>
$\log{\frac{Y}{1 - Y}} = \theta x^{T}$, "log odds"
- Log odds are linear in $X$
- This is called the logit of $Y$
  - Links the odds of $Y$ (a probability) to a linear regression in $X$
  - Logit ranges from -ve infite to +ve infinite
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

- $Y$ can be 1 or 0 (binary case)
- $Y \hspace{2 mm} | \hspace{2 mm} X$ ~ Bernoulli

----

## Logistic Regression
# Find parameters
<space>

- $Y$ can be 1 or 0 (binary case)
- $Y \hspace{2 mm} | \hspace{2 mm} X$ ~ Bernoulli
- $P(Y\hspace{2 mm} |\hspace{2 mm} X) = p$, when $Y$ = 1 
  - $p = h_{\theta}(x)$
- $P(Y\hspace{2 mm} |\hspace{2 mm} X) = 1-p$, when $Y$ = 0

----

## Logistic Regression
# Find parameters
<space>

- $Y$ can be 1 or 0 (binary case)
- $Y | X$ ~ Bernoulli
- $P(Y|X) = p$, when $Y$ = 1
  - $p = h_{\theta}(x)$
- $P(Y|X) = 1-p$, when $Y$ = 0
- $P(Y = y_{i}|X) = p^{y_{i}}(1-p)^{1-y_{i}}$
- Taking the log of both sides...

----

## Logistic Regression
# Find parameters
<space>

$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>

![plot of chunk cost_curves](figure/cost_curves1.png) ![plot of chunk cost_curves](figure/cost_curves2.png) 

----

## Logistic Regression
# Find parameters
<space>

$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>
- Logistic regression cost function is then<br>
$cost(h_{\theta}(x), y)  = \frac{1}{m} \sum_{i=1}^{m} -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$

----

## Logistic Regression
# Find parameters
<space>

$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>
- Logistic regression cost function is then<br>
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

- Class of hill-climbing techniques
- Efficient
- Easier to calculate that gradient descent
  - Except for first and second derivatives
- Fast

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume $f'(x_{0})$ is close to zero and $f''(x_{0})$ is positive

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume $f'(x_{0})$ is close to zero and $f''(x_{0})$ is positive
- Re-write $f(x)$ as its Taylor expansion:<br>
$f(x) = f(x_{0}) + (x-x_{0})f'(x_{0}) + \frac{1}{2}(x-x_{0})^{2}f''(x_{0})$

----

## Logistic Regression
# Newton-Raphson Method
<space>

- Assume $f'(x_{0})$ is close to zero and $f''(x_{0})$ is positive
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
[1,]         1    87.00
[2,]         2    58.00
[3,]         3    38.67
[4,]         4    25.78
[5,]         5    17.19
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
# Newton-Raphson Method
<space>

- Minimization algorithm
- Approximation, non-closed form solution
- Built-in to many programs
- Can be used to find the parameters of a logistic regression equation

----

## Logistic Regression
# Summary
<space>

- Very popular classification algorithm
- Part of family of GLMs
- Based on Binomial error terms, 1's and 0's
- Usually requires large sample size
- Assumes linearity between logit function and independent variables
- Uses sigmoid to link the probabilities with regression
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

![original_data](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/original_data.png)

----

## Principle Component Analysis
# Concepts
<space>

![calc_centroid](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/calc_centroid.png)

----

## Principle Component Analysis
# Concepts
<space>

![sub_mean](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/sub_mean.png)

----

## Principle Component Analysis
# Concepts
<space>

![max_var_dir](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/max_var_dir.png)

----

## Principle Component Analysis
# Concepts
<space>

![second_PC](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/second_PC.png)

----

## Principle Component Analysis
# Concepts
<space>

![rotated_grid](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/rotated_grid.png)

----

## Principle Component Analysis
# Concepts
<space>

![rotated_PCs](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/rotated_PCs.png)

----

## Principle Component Analysis
# Concepts
<space>

![new_axes](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/new_axes.png)

----

## Principle Component Analysis
# Concepts
<space>

![final_PC](/Users/ilanman/Desktop/Data/RPres_ML_2/figure/final_PC.png)

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
  - Provides most information

----

## Principle Component Analysis
# Digression...
<space>

- eigenvalues
- eigenvectors
- 

----

## Principle Component Analysis
# Assumptions
<space>

Eigenvectors and values exist in pairs: every eigenvector has a corresponding eigenvalue. An eigenvector is a direction, in the example above the eigenvector was the direction of the line (vertical, horizontal, 45 degrees etc.) . An eigenvalue is a number, telling you how much variance there is in the data in that direction, in the example above the eigenvalue is a number telling us how spread out the data is on the line. The eigenvector with the highest eigenvalue is therefore the principal component.

----

## Principle Component Analysis
# Eigenwhat?
<space>

$\bf{A}x = \lambda x$
  - $\lambda$ is an eigenvalue of $\bf{A}$ and $\bf{x}$ is an eigenvector of $\bf{A}$<br>
$\bf{A}x - \lambda Ix = 0$<br>
$(\bf{A} - \lambda I)x = 0$<br>
$\det(\bf{A} - \lambda I)$ = 0 &nbsp; <- roots of this yield eigenvalues of $\bf{A}$

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}, X = \begin{bmatrix} x_{1}\\ x_{2} \end{bmatrix}\]


----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}, X = \begin{bmatrix} x_{1}\\ x_{2} \end{bmatrix}\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X = \lambda X\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

\[A = \begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}, I= \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}, X = \begin{bmatrix} x_{1}\\ x_{2} \end{bmatrix}\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X = \lambda X\]
\[\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix}X - \lambda X = 0\]
\[(\begin{bmatrix} 5 & 2\\ 2 & 5 \end{bmatrix} - \lambda I)X = 0\]

----

## Principle Component Analysis
# Eigenwhat?
<space>

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
$\lambda = 3, 7$

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>
$x_{1} = -x_{2}$<br>

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 3$<br>
$Ax = 3x$<br>
$5x_{1} + 2x_{2} = 3x_{1}$<br>
$2x_{1} + 5x_{2} = 3x_{2}$<br>
$x_{1} = -x_{2}$<br>

\[Eigenvector = \begin{bmatrix} 1\\ -1 \end{bmatrix}\]

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 7$<br>
$Ax = 7x$<br>
$5x_{1} + 2x_{2} = 7x_{1}$<br>
$2x_{2} + 5x_{2} = 7x_{2}$<br>
$x_{1} = x_{2}$<br>

----

## Principle Component Analysis
# Eigencheck
<space>

- when $\lambda = 7$<br>
$Ax = 7x$<br>
$5x_{1} + 2x_{2} = 7x_{1}$<br>
$2x_{2} + 5x_{2} = 7x_{2}$<br>
$x_{1} = x_{2}$<br>

\[Eigenvector = \begin{bmatrix} 1\\ 1 \end{bmatrix}\]

----

## Principle Component Analysis
# Eigencheck
<space>

$Ax = \lambda x$

```r
x1 = c(1,-1)
x2 = c(1,1)
A %*% x1 == 3 * x1
A %*% x2 == 7 * x2
```

----

## Principle Component Analysis
# Eigencheck
<space>

$Ax = \lambda x$

```r
A %*% x1 == 3 * x1
```

```
     [,1]
[1,] TRUE
[2,] TRUE
```

```r
A %*% x2 == 7 * x2
```

```
     [,1]
[1,] TRUE
[2,] TRUE
```

----

## Principle Component Analysis
# Eigencheck
<space>

$A = \bf{x}\lambda\bf{x}^{T}$

```r
m <- matrix(c(x1,x2),ncol=2)
m <- m/sqrt(norm(m))  ## normalize
as.matrix(m %*% diag(roots) %*% t(m))
```

```
##      [,1] [,2]
## [1,]    5    2
## [2,]    2    5
```

----

## Principle Component Analysis
# Eigencheck
<space>

- Rewrite $A = \bf{x}\lambda\bf{x}$ into
$\lambda = \bf{x}^{T}A\bf{x}$
  - Diagonlization

----

## Principle Component Analysis
# Objective
<space>

- Covariance matrix<br>
$\bf{C} = \bf{XX}^{T}$
  - Diagonals are the variances, off-diagonals are covariances
  - Symmetric, positive semi-definite

- Restated goals are
  - Minimize covariance and maximize variance
  - Optimal $\bf{C}$ is a diagonal matrix, off diagonals are = 0

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
$=\bf{PX(PX)^{T}}$,  drop the $\frac{1}{n-1}$<br>
$=\bf{P(XX^{T})P^{T}}$,  because $(AB)^{T} = B^{T}A^{T}$<br>

----

## Principle Component Analysis
# Motivation
<space>

$\bf{PX} = \bf{Y}$<br>
$\bf{C_{Y}} = \frac{1}{(n-1)}\bf{YY^{T}}$<br>
$=\bf{PX(PX)^{T}}$, drop the $\frac{1}{n-1}$<br>
$=\bf{P(XX^{T})P^{T}}$,  because $(AB)^{T} = B^{T}A^{T}$<br> 
$=\bf{PAP^{T}}$<br>
- $\bf{P}$ is a matrix with columns that are eigenvectors
- $\bf{A}$ is a diagonalized matrix of eigenvalues and is symmetric, positive definite<br>
$\bf{A} = \bf{EDE^{T}}$

----

## Principle Component Analysis
# Motivation
<space>

- Each row of $\bf{P}$ should be an eigenvector of $\bf{A}$<br>
$\bf{P} = \bf{E^{T}}$<br>
Since $\bf{A} = \bf{EDE^{T}}$<br>

----

## Principle Component Analysis
# Motivation
<space>

- Each row of $\bf{P}$ should be an eigenvector of $\bf{A}$<br>
$\bf{P} = \bf{E^{T}}$<br>
Since $\bf{A} = \bf{EDE^{T}}$<br>
$\bf{A} = \bf{P^{T}DP}$<br>
$\bf{C_{Y}} = \bf{PAP^{T}}$<br>
$\bf{C_{Y}} = \bf{PP^{T}DPP^{T}}$<br> because $P^{T}=P^{-1}$ since $\bf{C_{Y}}$ is positive semi-definite
$= \frac{1}{n-1}\bf{D}$
- $\bf{D}$ is a diagonal matrix, depending on how we choose $\bf{P}$
- Therefore $\bf{C_{Y}}$ is diagonalized

----

## Principle Component Analysis
# Assumptions
<space>

- Assumes linear relationship between $\bf{X}$ and $\bf{Y}$ (non-linear is a kernel PCA)
- Orthogonal components - ensures no correlation among PCs
- Largest variance indicates most signal
- Assumes data is normally distributed, otherwise PCA might not diagonalize matrix
  - Can use ICA...
  - But most data is normal and PCA is robust to slight deviance from normality
  
----

## Principle Component Analysis
# Example
<space>


```r
data <- read.csv('tennis_data_2013.csv')
data$Player1 <- as.character(data$Player1)
data$Player2 <- as.character(data$Player2)

tennis <- data
m <- length(data)

for (i in 10:m){
  tennis[,i] <- ifelse(is.na(data[,i]),0,data[,i])
}

str(tennis)
```

```
## 'data.frame':	943 obs. of  35 variables:
##  $ unique_ID : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Tournament: Factor w/ 4 levels "AUS","FRE","USA",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ Gender    : Factor w/ 2 levels "F","M": 2 2 2 2 2 2 2 2 2 2 ...
##  $ Player1   : chr  "LukasLacko" "LeonardoMayer" "MarcosBaghdatis" "DmitryTursunov" ...
##  $ Player2   : chr  "NovakDjokovic" "AlbertMontanes" "DenisIstomin" "MichaelRussell" ...
##  $ Round     : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ Result    : int  0 1 0 1 0 0 0 1 0 1 ...
##  $ FNL1      : int  0 3 0 3 1 1 2 2 0 3 ...
##  $ FNL2      : int  3 0 3 0 3 3 3 0 3 2 ...
##  $ FSP.1     : int  61 61 52 53 76 65 68 47 64 77 ...
##  $ FSW.1     : int  35 31 53 39 63 51 73 18 26 76 ...
##  $ SSP.1     : int  39 39 48 47 24 35 32 53 36 23 ...
##  $ SSW.1     : int  18 13 20 24 12 22 24 15 12 11 ...
##  $ ACE.1     : num  5 13 8 8 0 9 5 3 3 6 ...
##  $ DBF.1     : num  1 1 4 6 4 3 3 4 0 4 ...
##  $ WNR.1     : num  17 13 37 8 16 35 41 21 20 6 ...
##  $ UFE.1     : num  29 1 50 6 35 41 50 31 39 4 ...
##  $ BPC.1     : num  1 7 1 6 3 2 9 6 3 7 ...
##  $ BPW.1     : num  3 14 9 9 12 7 17 20 7 24 ...
##  $ NPA.1     : num  8 0 16 0 9 6 14 6 5 0 ...
##  $ NPW.1     : num  11 0 23 0 13 12 30 9 14 0 ...
##  $ TPW.1     : num  70 80 106 104 128 108 173 78 67 162 ...
##  $ FSP.2     : int  68 60 77 50 53 63 60 54 67 60 ...
##  $ FSW.2     : int  45 23 57 24 59 60 66 26 42 68 ...
##  $ SSP.2     : int  32 40 23 50 47 37 40 46 33 40 ...
##  $ SSW.2     : int  17 9 15 19 32 22 34 13 14 25 ...
##  $ ACE.2     : num  10 1 9 1 17 24 2 0 12 8 ...
##  $ DBF.2     : num  0 4 1 8 11 4 6 11 0 12 ...
##  $ WNR.2     : num  40 1 41 1 59 47 57 11 32 8 ...
##  $ UFE.2     : num  30 4 41 8 79 45 72 46 20 12 ...
##  $ BPC.2     : num  4 0 4 1 3 4 10 2 7 6 ...
##  $ BPW.2     : num  8 0 13 7 5 7 17 6 10 14 ...
##  $ NPA.2     : num  8 0 12 0 16 14 25 8 8 0 ...
##  $ NPW.2     : num  9 0 16 0 28 17 36 12 11 0 ...
##  $ TPW.2     : num  101 42 126 79 127 122 173 61 94 141 ...
```

```r
features <- tennis[,10:m]

head(features)
```

```
##   FSP.1 FSW.1 SSP.1 SSW.1 ACE.1 DBF.1 WNR.1 UFE.1 BPC.1 BPW.1 NPA.1 NPW.1
## 1    61    35    39    18     5     1    17    29     1     3     8    11
## 2    61    31    39    13    13     1    13     1     7    14     0     0
## 3    52    53    48    20     8     4    37    50     1     9    16    23
## 4    53    39    47    24     8     6     8     6     6     9     0     0
## 5    76    63    24    12     0     4    16    35     3    12     9    13
## 6    65    51    35    22     9     3    35    41     2     7     6    12
##   TPW.1 FSP.2 FSW.2 SSP.2 SSW.2 ACE.2 DBF.2 WNR.2 UFE.2 BPC.2 BPW.2 NPA.2
## 1    70    68    45    32    17    10     0    40    30     4     8     8
## 2    80    60    23    40     9     1     4     1     4     0     0     0
## 3   106    77    57    23    15     9     1    41    41     4    13    12
## 4   104    50    24    50    19     1     8     1     8     1     7     0
## 5   128    53    59    47    32    17    11    59    79     3     5    16
## 6   108    63    60    37    22    24     4    47    45     4     7    14
##   NPW.2 TPW.2
## 1     9   101
## 2     0    42
## 3    16   126
## 4     0    79
## 5    28   127
## 6    17   122
```

```r
str(features)
```

```
## 'data.frame':	943 obs. of  26 variables:
##  $ FSP.1: int  61 61 52 53 76 65 68 47 64 77 ...
##  $ FSW.1: int  35 31 53 39 63 51 73 18 26 76 ...
##  $ SSP.1: int  39 39 48 47 24 35 32 53 36 23 ...
##  $ SSW.1: int  18 13 20 24 12 22 24 15 12 11 ...
##  $ ACE.1: num  5 13 8 8 0 9 5 3 3 6 ...
##  $ DBF.1: num  1 1 4 6 4 3 3 4 0 4 ...
##  $ WNR.1: num  17 13 37 8 16 35 41 21 20 6 ...
##  $ UFE.1: num  29 1 50 6 35 41 50 31 39 4 ...
##  $ BPC.1: num  1 7 1 6 3 2 9 6 3 7 ...
##  $ BPW.1: num  3 14 9 9 12 7 17 20 7 24 ...
##  $ NPA.1: num  8 0 16 0 9 6 14 6 5 0 ...
##  $ NPW.1: num  11 0 23 0 13 12 30 9 14 0 ...
##  $ TPW.1: num  70 80 106 104 128 108 173 78 67 162 ...
##  $ FSP.2: int  68 60 77 50 53 63 60 54 67 60 ...
##  $ FSW.2: int  45 23 57 24 59 60 66 26 42 68 ...
##  $ SSP.2: int  32 40 23 50 47 37 40 46 33 40 ...
##  $ SSW.2: int  17 9 15 19 32 22 34 13 14 25 ...
##  $ ACE.2: num  10 1 9 1 17 24 2 0 12 8 ...
##  $ DBF.2: num  0 4 1 8 11 4 6 11 0 12 ...
##  $ WNR.2: num  40 1 41 1 59 47 57 11 32 8 ...
##  $ UFE.2: num  30 4 41 8 79 45 72 46 20 12 ...
##  $ BPC.2: num  4 0 4 1 3 4 10 2 7 6 ...
##  $ BPW.2: num  8 0 13 7 5 7 17 6 10 14 ...
##  $ NPA.2: num  8 0 12 0 16 14 25 8 8 0 ...
##  $ NPW.2: num  9 0 16 0 28 17 36 12 11 0 ...
##  $ TPW.2: num  101 42 126 79 127 122 173 61 94 141 ...
```

```r
dim(features)
```

```
## [1] 943  26
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
sum(diag(Cy) - eigenvalues)
```

```
## [1] 1.416e-14
```

```r
sum(Cy[(upper.tri(Cy)|lower.tri(Cy))])   ## off diagonals are 0 since PC's are orthogonal
```

```
## [1] -2.041e-15
```

----

## Principle Component Analysis
# Example
<space>

![plot of chunk var_expl_plot](figure/var_expl_plot.png) 


----

## Principle Component Analysis
# Example
<space>


```r
pca.df <- prcomp(scaled_features)
round(eigenvalues,10) == round((pca.df$sdev)^2,10)
```

```
##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [15] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
```

```r
round(eigenvectors[,1],10) == round(pca.df$rotation[,1],10)
```

```
## FSP.1 FSW.1 SSP.1 SSW.1 ACE.1 DBF.1 WNR.1 UFE.1 BPC.1 BPW.1 NPA.1 NPW.1 
##  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE 
## TPW.1 FSP.2 FSW.2 SSP.2 SSW.2 ACE.2 DBF.2 WNR.2 UFE.2 BPC.2 BPW.2 NPA.2 
##  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE 
## NPW.2 TPW.2 
##  TRUE  TRUE
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

![plot of chunk tennis_plot_gender](figure/tennis_plot_gender.png) 

----

## Principle Component Analysis
# Example
<space>

- Classify based on PC1?


```r
gen <- ifelse(pca.df$x[,1] > abs(mean(pca.df$x[,1]))*2,"F","M")
sum(diag(table(gen,as.character(data$Gender))))/rows
```

```
## [1] 0.7646
```

----

## Principle Component Analysis
# Summary
<space>

- Very popular dimensionality reduction technique
- Intuitive
- Cannot reverse engineer dataset easily
- Sparse PCA emphasizes important features
- Non-linear structure is difficult to model with PCA
- Extensions (ICA, kernel PCA) developed to generalize

----

## Clustering
# Motivation
<space>

- Separate data into meaningful or useful groups
  - Capture natural structure of the data
  - Starting point for further analysis
- Cluster for utility
  - Summarizing data for less expensive computation
  - Data compression

----

## Clustering
# Types of Clusters
<space>

- Data that looks similar
- Prototype based
- Density based

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

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 

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

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 

----

## Clustering
# DBSCAN
<space>

- A cluster is a dense region of points separated by low-density regions
- Group objects into one cluster if they are connected to one another by densely populated area
- Used when the clusters are irregular or intertwined, and when noise and outliers are present
- 

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

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

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
  - Dependent on values of K, Eps
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
    democrat |         83 |          1 |         84 | 
             |      0.572 |      0.007 |            | 
-------------|------------|------------|------------|
  republican |          5 |         56 |         61 | 
             |      0.034 |      0.386 |            | 
-------------|------------|------------|------------|
Column Total |         88 |         57 |        145 | 
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
## physician-fee-freeze                97.58
## synfuels-corporation-cutback        39.79
## mx-missile                           9.69
## adoption-of-the-budget-resolution    7.96
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
## C5.0 [Release 2.07 GPL Edition]  	Mon Oct  6 22:38:45 2014
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 289 cases (17 attributes) from undefined.data
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (172.2/1.6)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (95.4/3.7)
##     synfuels-corporation-cutback = y:
##     :...mx-missile = ?: republican (0)
##         mx-missile = y: democrat (2.5)
##         mx-missile = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (13.9/2.3)
##             adoption-of-the-budget-resolution = y: democrat (5/1)
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
## 	   178     5    (a): class democrat
## 	     3   103    (b): class republican
## 
## 
## 	Attribute usage:
## 
## 	 97.58%	physician-fee-freeze
## 	 39.79%	synfuels-corporation-cutback
## 	  9.69%	mx-missile
## 	  7.96%	adoption-of-the-budget-resolution
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
##     democrat |         83 |          1 |         84 | 
##              |      0.572 |      0.007 |            | 
## -------------|------------|------------|------------|
##   republican |          6 |         55 |         61 | 
##              |      0.041 |      0.379 |            | 
## -------------|------------|------------|------------|
## Column Total |         89 |         56 |        145 | 
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
## C5.0 [Release 2.07 GPL Edition]  	Mon Oct  6 22:38:45 2014
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
## physician-fee-freeze in {?,n}: democrat (172.2/1.6)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (95.4/3.7)
##     synfuels-corporation-cutback = y:
##     :...mx-missile = ?: republican (0)
##         mx-missile = y: democrat (2.5)
##         mx-missile = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (13.9/2.3)
##             adoption-of-the-budget-resolution = y: democrat (5/1)
## 
## -----  Trial 1:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (142.9/14.2)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (56/3.4)
##     immigration = n:
##     :...education-spending in {?,n}: democrat (35.7/5.4)
##         education-spending = y: republican (54.4/20.1)
## 
## -----  Trial 2:  -----
## 
## Decision tree:
## 
## adoption-of-the-budget-resolution in {?,y}: democrat (156.8/23.2)
## adoption-of-the-budget-resolution = n:
## :...synfuels-corporation-cutback = ?: republican (0)
##     synfuels-corporation-cutback = y: democrat (39.8/13.9)
##     synfuels-corporation-cutback = n:
##     :...duty-free-exports in {?,n}: republican (77.7/3.4)
##         duty-free-exports = y: democrat (14.6/3.9)
## 
## -----  Trial 3:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (105.2/14.5)
## physician-fee-freeze = y:
## :...water-project-cost-sharing in {?,n}: republican (69.3/7.3)
##     water-project-cost-sharing = y:
##     :...duty-free-exports = ?: republican (0)
##         duty-free-exports = y: democrat (11.7/1)
##         duty-free-exports = n:
##         :...adoption-of-the-budget-resolution in {?,n}: republican (73.7/16.6)
##             adoption-of-the-budget-resolution = y: democrat (29.1/9.7)
## 
## -----  Trial 4:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (95.5/20.7)
## physician-fee-freeze = y:
## :...immigration in {?,y}: republican (81/14.2)
##     immigration = n:
##     :...anti-satellite-test-ban in {?,y}: republican (17.7/0.4)
##         anti-satellite-test-ban = n:
##         :...adoption-of-the-budget-resolution = n: republican (71.9/30.8)
##             adoption-of-the-budget-resolution in {?,y}: democrat (23/1.5)
## 
## -----  Trial 5:  -----
## 
## Decision tree:
## 
## duty-free-exports in {?,y}: democrat (85.4/14.6)
## duty-free-exports = n:
## :...synfuels-corporation-cutback = ?: republican (0)
##     synfuels-corporation-cutback = y:
##     :...water-project-cost-sharing = n: republican (15.5/3.9)
##     :   water-project-cost-sharing in {?,y}: democrat (77.8/19.4)
##     synfuels-corporation-cutback = n:
##     :...crime = ?: republican (0)
##         crime = n: democrat (5.8/0.4)
##         crime = y:
##         :...religious-groups-in-schools in {?,n}: republican (23.1/0.7)
##             religious-groups-in-schools = y:
##             :...physician-fee-freeze = n: democrat (9.3/0.5)
##                 physician-fee-freeze in {?,y}: republican (72.1/13.7)
## 
## -----  Trial 6:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (92.2/20.2)
## physician-fee-freeze = y:
## :...religious-groups-in-schools in {?,n}: republican (9.7/0.2)
##     religious-groups-in-schools = y:
##     :...el-salvador-aid = ?: republican (0)
##         el-salvador-aid = n: democrat (17.4/2.2)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban in {?,y}: republican (31)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution in {?,
##                 :                                     n}: republican (91.4/34.1)
##                 adoption-of-the-budget-resolution = y: democrat (47.3/13.2)
## 
## -----  Trial 7:  -----
## 
## Decision tree:
## 
## synfuels-corporation-cutback in {?,y}: democrat (132.2/39.1)
## synfuels-corporation-cutback = n:
## :...crime = ?: republican (0)
##     crime = n: democrat (14.8/1.2)
##     crime = y:
##     :...immigration = n: democrat (57.5/26.7)
##         immigration in {?,y}: republican (84.5/11.1)
## 
## -----  Trial 8:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n:
## :...adoption-of-the-budget-resolution = n: republican (26.7/9.3)
## :   adoption-of-the-budget-resolution in {?,y}: democrat (52.2/3.6)
## physician-fee-freeze = y:
## :...water-project-cost-sharing in {?,n}: republican (62.6/8.5)
##     water-project-cost-sharing = y:
##     :...duty-free-exports = ?: republican (0)
##         duty-free-exports = y: democrat (20.9/2.1)
##         duty-free-exports = n:
##         :...synfuels-corporation-cutback in {?,n}: republican (53.8/9.2)
##             synfuels-corporation-cutback = y: democrat (72.9/31.4)
## 
## -----  Trial 9:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (97.2/17.9)
## physician-fee-freeze = y:
## :...education-spending = n: democrat (76.3/33.4)
##     education-spending in {?,y}: republican (115.5/27.1)
## 
## -----  Trial 10:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (85.8/19.3)
## physician-fee-freeze = y:
## :...water-project-cost-sharing in {?,n}: republican (57.1/8.7)
##     water-project-cost-sharing = y:
##     :...duty-free-exports = ?: republican (0)
##         duty-free-exports = y: democrat (16/1.3)
##         duty-free-exports = n:
##         :...el-salvador-aid = ?: republican (0)
##             el-salvador-aid = n: democrat (5.9/0.4)
##             el-salvador-aid = y:
##             :...superfund-right-to-sue = n: democrat (19.2/4.7)
##                 superfund-right-to-sue in {?,y}: republican (105/28.6)
## 
## -----  Trial 11:  -----
## 
## Decision tree:
## 
## synfuels-corporation-cutback in {?,y}: democrat (135.7/41.6)
## synfuels-corporation-cutback = n:
## :...crime = n: democrat (8.2/0.5)
##     crime in {?,y}: republican (145/53.7)
## 
## -----  Trial 12:  -----
## 
## Decision tree:
## 
## adoption-of-the-budget-resolution = ?: democrat (0)
## adoption-of-the-budget-resolution = y:
## :...physician-fee-freeze in {?,n}: democrat (37/1.9)
## :   physician-fee-freeze = y:
## :   :...superfund-right-to-sue = n: republican (17.3/2.2)
## :       superfund-right-to-sue in {?,y}: democrat (77.7/20.8)
## adoption-of-the-budget-resolution = n:
## :...duty-free-exports = ?: republican (0)
##     duty-free-exports = y: democrat (29.9/9.1)
##     duty-free-exports = n:
##     :...religious-groups-in-schools in {?,n}: republican (21.7)
##         religious-groups-in-schools = y:
##         :...physician-fee-freeze = ?: republican (0)
##             physician-fee-freeze = n: democrat (15.5/0.8)
##             physician-fee-freeze = y:
##             :...superfund-right-to-sue = n: democrat (12.4/2.5)
##                 superfund-right-to-sue in {?,y}: republican (77.5/16.6)
## 
## -----  Trial 13:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (69.1/17)
## physician-fee-freeze = y:
## :...water-project-cost-sharing in {?,n}: republican (77.3/10.7)
##     water-project-cost-sharing = y:
##     :...duty-free-exports = ?: republican (0)
##         duty-free-exports = y: democrat (12.9/1.1)
##         duty-free-exports = n:
##         :...export-administration-act-south-africa = n: democrat (44.2/15.3)
##             export-administration-act-south-africa in {?,
##                                                        y}: republican (85.6/30.1)
## 
## -----  Trial 14:  -----
## 
## Decision tree:
## 
## adoption-of-the-budget-resolution = ?: republican (0)
## adoption-of-the-budget-resolution = n:
## :...immigration in {?,y}: republican (78.5/12.6)
## :   immigration = n:
## :   :...physician-fee-freeze = n: democrat (11.7/0.6)
## :       physician-fee-freeze in {?,y}: republican (74.2/28.7)
## adoption-of-the-budget-resolution = y:
## :...physician-fee-freeze in {?,n}: democrat (25.5/1.3)
##     physician-fee-freeze = y:
##     :...anti-satellite-test-ban in {?,n}: democrat (73.6/22.7)
##         anti-satellite-test-ban = y: republican (25.5/3.5)
## 
## -----  Trial 15:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (72.3/14.7)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (122.9/31.1)
##     synfuels-corporation-cutback = y:
##     :...water-project-cost-sharing = n: republican (9.2/0.6)
##         water-project-cost-sharing in {?,y}: democrat (84.6/27.1)
## 
## -----  Trial 16:  -----
## 
## Decision tree:
## 
## duty-free-exports = ?: republican (0)
## duty-free-exports = y: democrat (75.3/21)
## duty-free-exports = n:
## :...religious-groups-in-schools in {?,n}: republican (21.3/2.4)
##     religious-groups-in-schools = y:
##     :...physician-fee-freeze = ?: republican (0)
##         physician-fee-freeze = n: democrat (18.5/1)
##         physician-fee-freeze = y:
##         :...el-salvador-aid = ?: republican (0)
##             el-salvador-aid = n: democrat (9/1.3)
##             el-salvador-aid = y:
##             :...anti-satellite-test-ban in {?,y}: republican (18.7)
##                 anti-satellite-test-ban = n:
##                 :...aid-to-nicaraguan-contras = ?: republican (0)
##                     aid-to-nicaraguan-contras = y: democrat (8.6/0.7)
##                     aid-to-nicaraguan-contras = n:
##                     :...immigration = n: democrat (77.2/35)
##                         immigration in {?,y}: republican (60.4/2.6)
## 
## -----  Trial 17:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (67.9/14.5)
## physician-fee-freeze = y:
## :...education-spending in {?,y}: republican (132.7/24.8)
##     education-spending = n:
##     :...superfund-right-to-sue = n: republican (18.4/2.4)
##         superfund-right-to-sue in {?,y}: democrat (70/25.7)
## 
## -----  Trial 18:  -----
## 
## Decision tree:
## 
## adoption-of-the-budget-resolution = ?: republican (0)
## adoption-of-the-budget-resolution = y: democrat (109.9/34.8)
## adoption-of-the-budget-resolution = n:
## :...duty-free-exports = ?: republican (0)
##     duty-free-exports = y: democrat (30.7/11.6)
##     duty-free-exports = n:
##     :...religious-groups-in-schools in {?,n}: republican (20.8)
##         religious-groups-in-schools = y:
##         :...physician-fee-freeze = ?: republican (0)
##             physician-fee-freeze = n: democrat (7.6/0.3)
##             physician-fee-freeze = y:
##             :...mx-missile in {?,n}: republican (116.6/20.1)
##                 mx-missile = y: democrat (3.3/0.1)
## 
## -----  Trial 19:  -----
## 
## Decision tree:
## 
## physician-fee-freeze = ?: republican (0)
## physician-fee-freeze = n: democrat (49/13.4)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (142.7/25.3)
##     synfuels-corporation-cutback = y:
##     :...aid-to-nicaraguan-contras = n: republican (82.5/37.3)
##         aid-to-nicaraguan-contras in {?,y}: democrat (14.7/0.5)
## 
## -----  Trial 20:  -----
## 
## Decision tree:
## 
## synfuels-corporation-cutback = ?: republican (0)
## synfuels-corporation-cutback = y: democrat (122.8/38.7)
## synfuels-corporation-cutback = n:
## :...crime = n: democrat (7.3/0.2)
##     crime in {?,y}: republican (158.9/45.4)
## 
## -----  Trial 21:  -----
## 
## Decision tree:
## 
## immigration = ?: democrat (0)
## immigration = y:
## :...physician-fee-freeze = n: democrat (38.8/13.8)
## :   physician-fee-freeze in {?,y}: republican (93.6/20.4)
## immigration = n:
## :...adoption-of-the-budget-resolution in {?,y}: democrat (63.3/8.5)
##     adoption-of-the-budget-resolution = n:
##     :...superfund-right-to-sue = n: democrat (16.9/0.6)
##         superfund-right-to-sue in {?,y}: republican (76.5/28.8)
## 
## -----  Trial 22:  -----
## 
## Decision tree:
## 
## water-project-cost-sharing = ?: democrat (0)
## water-project-cost-sharing = n:
## :...physician-fee-freeze = n: democrat (43.2/16)
## :   physician-fee-freeze in {?,y}: republican (62.8/10.4)
## water-project-cost-sharing = y:
## :...duty-free-exports in {?,y}: democrat (34.4/1.1)
##     duty-free-exports = n:
##     :...el-salvador-aid in {?,n}: democrat (10.5/0.3)
##         el-salvador-aid = y:
##         :...anti-satellite-test-ban = ?: democrat (0)
##             anti-satellite-test-ban = y: republican (12.3)
##             anti-satellite-test-ban = n:
##             :...adoption-of-the-budget-resolution = n: republican (78.6/32.3)
##                 adoption-of-the-budget-resolution in {?,
##                                                       y}: democrat (47.2/7.4)
## 
## -----  Trial 23:  -----
## 
## Decision tree:
## 
## synfuels-corporation-cutback in {?,y}: democrat (128.2/35.6)
## synfuels-corporation-cutback = n:
## :...immigration = n: democrat (82.3/31.9)
##     immigration in {?,y}: republican (78.5/16.9)
## 
## -----  Trial 24:  -----
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (83.2/15.7)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (110.1/30.8)
##     synfuels-corporation-cutback = y:
##     :...water-project-cost-sharing = n: republican (8.3/0.3)
##         water-project-cost-sharing in {?,y}: democrat (86.3/28.4)
## 
## 
## Evaluation on training data (289 cases):
## 
## Trial	    Decision Tree   
## -----	  ----------------  
## 	  Size      Errors  
## 
##    0	     5    8( 2.8%)
##    1	     4   19( 6.6%)
##    2	     4   32(11.1%)
##    3	     5    9( 3.1%)
##    4	     5   11( 3.8%)
##    5	     7   30(10.4%)
##    6	     6   11( 3.8%)
##    7	     4   68(23.5%)
##    8	     6   26( 9.0%)
##    9	     3   20( 6.9%)
##   10	     6   11( 3.8%)
##   11	     3   40(13.8%)
##   12	     8   18( 6.2%)
##   13	     5   23( 8.0%)
##   14	     6   17( 5.9%)
##   15	     4   13( 4.5%)
##   16	     8   66(22.8%)
##   17	     4   14( 4.8%)
##   18	     6   19( 6.6%)
##   19	     4   11( 3.8%)
##   20	     3   41(14.2%)
##   21	     5   19( 6.6%)
##   22	     7   13( 4.5%)
##   23	     3   93(32.2%)
##   24	     4   13( 4.5%)
## boost	          4( 1.4%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	   182     1    (a): class democrat
## 	     3   103    (b): class republican
## 
## 
## 	Attribute usage:
## 
## 	 98.27%	immigration
## 	 97.92%	adoption-of-the-budget-resolution
## 	 97.58%	physician-fee-freeze
## 	 95.16%	synfuels-corporation-cutback
## 	 93.43%	duty-free-exports
## 	 88.93%	water-project-cost-sharing
## 	 62.28%	religious-groups-in-schools
## 	 61.94%	crime
## 	 47.40%	el-salvador-aid
## 	 42.21%	anti-satellite-test-ban
## 	 40.83%	superfund-right-to-sue
## 	 38.75%	education-spending
## 	 32.87%	mx-missile
## 	 31.14%	aid-to-nicaraguan-contras
## 	 20.07%	export-administration-act-south-africa
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
##     democrat |         78 |          6 |         84 | 
##              |      0.538 |      0.041 |            | 
## -------------|------------|------------|------------|
##   republican |          1 |         60 |         61 | 
##              |      0.007 |      0.414 |            | 
## -------------|------------|------------|------------|
## Column Total |         79 |         66 |        145 | 
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
