---

title       : Machine Learning with R - II
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
5. Missing data

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

x <- 1:10
log_ex <- data.frame(Y=c(rnorm(5,0,0.01),rnorm(5,5,0.01)),X=x)
ggplot(log_ex,aes(X,Y)) + geom_point(color='blue',size=3) + stat_smooth(method='lm',se=F,color='green',size=1)

----

## Logistic Regression
# Motivation
<space>


```r
library("MASS")
library(ggplot2)
data(menarche)
log_data <- data.frame(Y=menarche$Menarche/menarche$Total)
log_data$X <- menarche$Age

glm.out <- glm(cbind(Menarche, Total-Menarche) ~ Age,family=binomial(logit), data=menarche)
lm.out <- lm(Y~X, data=log_data)

log_data$fitted <- glm.out$fitted

data_points <- ggplot(log_data) + geom_point(aes(x=X,y=Y),color='blue',size=3)
line_points <- data_points + geom_abline(intercept = coef(lm.out)[1], slope = coef(lm.out)[2],color='green',size=1)
curve_points <- line_points + geom_line(aes(x=X,y=fitted),color='red',size=1) 
```

----

## Logistic Regression
# Notation
<space>

- type of regression to predict the probability of being in a class
  - typical to set threshold to 0.5
- assumes error terms are Binomially distributed
  - which generates 1's and 0's as the error term
- sigmoid or logistic function: $g(z) = \frac{1}{1+e^{-z}}$
  - interpret the output as $P(Y=1 | X)$
  - bounded by 0 and 1


```r
curve(1/(1+exp(-x)), from = -10, to = 10, ylab="P(Y=1|X)", col = 'red', lwd = 3.0)
abline(a=0.5, b=0, lty=2, col='blue', lwd = 3.0)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 

----

## Logistic Regression
# Find parameters
<space>

- The hypothesis function, $h_{\theta}(x)$, is P(Y=1|X)
- Linear Regression --> $h_{\theta}(x) = \theta x^{T}$
- Logistic Regression --> $h_{\theta}(x) = g(\theta x^{T})$ 
<br>
where $g(z) = \frac{1}{1+e^{-z}}$

----

## Logistic Regression
# Motivation
<space>

- Re-arranging $Y = \frac{1}{1+e^{-\theta x^{T}}}$ yields
<br>
<br>
$\log{\frac{Y}{1 - Y}} = \theta x^{T}$<br>
- log odds are linear in X
- this is called the logit of theta - this is linear in X

----

## Logistic Regression
# Find parameters
<space>

- So $h_{\theta}(x) = \frac{1}{1+e^{-\theta x^{T}}}$
- What is the cost function?
- Why can't we use the same cost function as before?
  - logistic residuals are Binomially distributed - not NORMAL
  - the regression function is not linear in X


----

## Logistic Regression
# Find parameters
<space>

- Define cost function as:

$cost(h_{\theta}(x)):$<br>
$= -\log(x),   y = 1$<br>
$= -\log(1-x),   y = 0$

![plot of chunk cost_curves](figure/cost_curves1.png) ![plot of chunk cost_curves](figure/cost_curves2.png) 

----

## Logistic Regression
# Find parameters
<space>

- using statistics, it can be shown that<br>
$cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$<br>
- Logistic regression cost function is then<br>
$cost(h_{\theta}(x), y)  = \frac{1}{m} \sum_{i=1}^{m} -y \log(h_{\theta}(x)) + (1-y) \log(1-h_{\theta}(x))$
- Minimize the cost

----

## Regression example
# Gradient descent
<space>

![plot of chunk grad_ex_plot](figure/grad_ex_plot.png) 

----

## Regression example
# Gradient descent
<space>


```r
x <- cbind(1,x)  #Add ones to x  
theta<- c(0,0)  # initalize theta vector 
m <- nrow(x)  # Number of the observations 
grad_cost <- function(X,y,theta) return(sum(((X%*%theta)- y)^2))
```

----

## Regression example
# Gradient descent
<space>


```r
gradDescent<-function(X,y,theta,iterations,alpha){
  m <- length(y)
  grad <- rep(0,length(theta))
  cost.df <- data.frame(cost=0,theta=0)
  
  for (i in 1:iterations){
    h <- X%*%theta
    grad <-  (t(X)%*%(h - y))/m
    theta <- theta - alpha * grad
    cost.df <- rbind(cost.df,c(grad_cost(X,y,theta),theta))    
  }  
  
  return(list(theta,cost.df))
}
```

----

## Regression example
# Gradient descent
<space>


```r
## initialize X, y and theta
X1<-matrix(ncol=1,nrow=nrow(df),cbind(1,df$X))
Y1<-matrix(ncol=1,nrow=nrow(df),df$Y)

init_theta<-as.matrix(c(0))
grad_cost(X1,Y1,init_theta)
```

```
[1] 5296
```

```r
iterations = 100
alpha = 0.1
results <- gradDescent(X1,Y1,init_theta,iterations,alpha)
```

----

## Regression example
# Gradient descent
<space>


```
## Error: object 'cost.df' not found
```

----

## Regression example
# Gradient descent
<space>


```r
grad_cost(X1,Y1,theta[[1]])
```

```
[1] 331.2
```

```r
## Make some predictions
intercept <- df[df$X==0,]$Y
pred <- function (x) return(intercept+c(x)%*%theta)
new_points <- c(0.1,0.5,0.8,1.1)
new_preds <- data.frame(X=new_points,Y=sapply(new_points,pred))
```

----

## Regression example
# Gradient descent
<space>


```r
ggplot(data=df,aes(x=X,y=Y))+geom_point(size=2)
```

![plot of chunk new_point](figure/new_point1.png) 

```r
ggplot(data=df,aes(x=X,y=Y))+geom_point()+geom_point(data=new_preds,aes(x=X,y=Y,color='red'),size=3)+scale_colour_discrete(guide = FALSE)
```

![plot of chunk new_point](figure/new_point2.png) 

----

## Regression example
# Gradient descent - summary
<space>

- minimization algorithm
- approximation, non-closed form solution
- good for large number of examples
- hard to select the right $\alpha$
- traditional looping is slow - optimization algorithms are used in practice

----

## Logistic Regression
# Summary
<space>

- very popular classification algorithm
- based on Binomial error terms, i.e. 1's and 0's
- 

----

## Principle Component Analysis
# Motivation
<space>

- used widely in modern data analysis
- not well understood
- intuition: reduce data into only relevant dimensions
- the goal of PCA is to compute the most meaningful was to re-express noisy data, revealing the hidden structure

----

## Principle Component Analysis
# Concepts
<space>

- first big assumption: linearity
- $PX=Y$
  - $X$ is original dataset, $P$ is a transformation of $X$ into $Y$
- how do we choose $P$?
  - reduce noise
  - maximize variance

----

## Principle Component Analysis
# Concepts
<space>

- covariance matrix
     - $C = X*X^{T}$

- restated goals are
  - minimize covariance and maximize variance
  - the optimizal $C$ is a diagonal matrix, off diagonals are = 0
  
----

## Principle Component Analysis
# Concepts
<space>

- summary of assumptions
  - linearity (non-linear is a kernel PCA)
  - largest variance indicates most signal, low variance = noise
  - orthogonal components - makes the linear algebra easier
  - assumes data is normally distributed, otherwise PCA might not diagonalize matrix
    - can use ICA
    - but most data is normal and PCA is robust to slight deviance from normality

----

## Principle Component Analysis
# Eigenwhat?
<space>

- $Ax = \lambdax$
  - $\lambda$ is an eigenvalue of $A$ and $x$ is an eigenvector of $A$
- $Ax - \lambdaIx = 0$
- $(A - \lambdaI)x = 0$
- $\det(A - \lambdaI)$ = 0


----

## Principle Component Analysis
# Eigenwhat?
<space>

$\[A=\left[{\begin{array}{cc}5 & 2 \\2 & 5\\\end{array}\right ]\]$

A = matrix(c(5,2,2,5),nrow=2)
I = diag(nrow(A))
|A - L*I| = 0
det(c(5-l,2,2,5-l))
(5-l)*(5-l) - 4 = 0
25 - 10l + l^2 - 4 = 0
l^2 - 10l + 21 = 0
roots <- Re(polyroot(c(21,-10,1)))
```

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when lambda = -3
Ax = 3x
5x1 + 2x2 = 3x1
2x1 + 5x2 = 3x2
x1=-x2
- one eigenvector = [1 -1]

----

## Principle Component Analysis
# Eigenwhat?
<space>

- when lambda = 7
5x1 + 2x2 = 7x1
2x2 + 5x2 = 7x2
x1 = x2
- another eigenvector = [1 1]

----

## Principle Component Analysis
# Eigenwhat?
<space>

A%*%c(1,-1) == 3 * as.matrix(c(1,-1))
A%*%c(1,1) == 7 * as.matrix(c(1,1))
roots

----

## Principle Component Analysis
# Eigenwhat?
<space>

- check
m <- matrix(c(1,-1,1,1),ncol=2)
m <- m/sqrt(norm(m))
as.matrix(m%*%diag(roots)%*%t(m))
- lambda is a diagonal matrix, with 0 off diagonals

----

## Principle Component Analysis
# Motivation
<space>

PX = Y

CY = (1/(n-1))*YYt
=PX(PX)t
=PXXtPt
=PAPt
# P is a matrix with columns that are eigenvectors
# A is a diagonalized matrix of eigenvalues (by linear algebra) and symmetric
A = EDEt

----

## Principle Component Analysis
# Motivation
<space>

# each row of P should be an eigenvector of A
P=Et
# also note that Pt = P-1 (linear algebra)
A = PtDP
CY = PPtDPPt
= (1/(n-1))*D
# D is a diagonal matrix, depending on how we choose P
# therefore CY is diagonalized

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

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 

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

- used to separate data into meaningful or useful groups (or both)
     - capture natural structure of the data
     - useful starting point for further analysis
- customer segmentation
- cluster for utility
     - summarizing data for less expensive computation
     - data compression
     - nearest neighbors - distance between two cluster centers (centroids)

----

## Clustering
# Motivation
<space>

- types of clusters
     - data points that are more similar to one another than points outside of the cluster - most intuitive definition
     - prototype-based: each data point is more similar to the prototype, i.e. center, of the cluster than the prototype of other clusters. Often a centroid, i.e. mean.
     - density based clusters: where the density is highest, that is a cluster. Works well for data with noise and outliers. Clusters separated by noise.

----

## Clustering
# Motivation
<space>

- example of density based cluster



----

## Clustering
# Kmeans
<space>

- k-means
     - prototype, partitional based
     - choose K initial centroids/clusters
     - points are assigned to the closest centroid
     - centroid is then updated based on the points in that cluster
     - update steps until no point changes or centroids remain the same

----

## Clustering
# Kmeans
<space>

1. Select K points as initial centroids. 
2. repeat
3.     Form K clusters by assigning each point to its closest centroid.
4.     Recompute the centroid of each cluster. 
5. until Centroids do not change, or change very minimally, i.e. <1%


3. Use similarity measures such as Euclidean or cosine similarity depending on the data
4. Minimize the squared distance of each point to closest centroid, minimize the objective function
     - the centroid that minimizes the SSE of the cluster is the mean
     - Kmeans leads to local minimum, not global, since you’re optimizing based on the centroids you chose, not all possible centroids

----

## Clustering
# Kmeans
<space>

- choose K randomly - can lead to poor centroids
     - run k-means multiple times - still doesn’t solve problems

- can reduce the total SSE by increasing the K
     - can increase the cluster with largest SSE
- can decrease K and minimize SSE
     - split up a cluster into other clusters. the centroid that is split will increase total SSE the least
- bisecting K means
     - less susceptible to initialization problems
     - split points into 2 clusters
          - take cluster with largest SSE - split that into two clusters
     - rerun bisecting K mean on resulting clusters
     - stop when you have K clusters

----

## Clustering
# Kmeans
<space>

- K mean fails
     - if some clusters are much bigger than other clusters - it cannot distinguish between natural clusters
     - if clusters have different densities, K means cannot tell 
     - distance metric doesn’t account for non-globular clusters, i.e. if they follow a distribution
- K means will still work if user accepts sub clusters of the natural cluster
- strengths
     - simple, efficient computationally
     - not useful for non-globular, different density, different sized data
     - outlier detection and removal can help address outlier problem
- can derive K mean algorithm using gradient descent
     - can use calculus to show that the mean of the cluster is the best choice of centroid, i.e. minimizes SSE

----

## Clustering
# Kmeans
<space>

tennis_kmean <- kmeans(features, centers=5)

# K MEANS DOES A GOOD JOB IN CLUSTERING GENDERS
table(tennis$Gender,tennis_kmean$cluster)

----

## Clustering
# Kmeans animation
<space>

# animate kmean algorithm
install.packages('animation')
library(animation)

oopt = ani.options(interval = 1)
## the kmeans() example; very fast to converge!
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
# DBSCAN
<space>

- density based
     - center based approach to finding density
     - count the number of points within some radius of a point, the radius is call Eps
     - if Eps is too big, there will be m points, if eps is too small, there will be 1 point
     - core point has X points within a radius of Eps, border points are within a radius of Eps of core point, and noise points are not within Eps of border or core points
     - if p is density connected to q, they are part of the same cluster, if not, then they are not; if p is not density connected to any other point, its considered noise
     
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

----

## Trees
# Motivation
<space>

- A root node that has no incoming edges and zero or more outgoing edges.
- Internal nodes, each of which has exactly one incoming edge and two or more outgoing edges
- Leaf or terminal nodes, each of which has exactly one incoming edge and no outgoing edges. 

- The non- terminal nodes, which include the root and other internal nodes, contain attribute test conditions to separate records that have different characteristics

- trees work best with categorical values
- descriptions are disjoint
- trees are robust to data errors
- training data is missing values

----

## Trees
# Algorithm(s)
<space>

ID3
C4.5
C5.0

----

## Trees
# Entropy calculation
<space>

- The entropy of a sample of data indicates how mixed the class values are; the minimum value of 0 indicates that the sample is completely homogenous, while 1 indicates the maximum amount of disorder.

entropy_function <- function(p) {

  if (min(p) < 0 || sum(p) <= 0) {
    return(NA)
  } else {
    p.norm <- p[p>0]/sum(p)
    -sum(log2(p.norm)*p.norm)
    }
}

----

## Trees
# Entropy calculation
<space>

- InfoGain = Entropy (pre split) - Entropy (post split)
     - Entropy is weighted by the Entropy of each feature split
 - avoid pre-pruning because its impossible to know if the tree will miss subtle but important patterns in the data (if you prune too early)
- hard to know optimal length of tree without growing it there first

----

## Trees
# Entropy calculation
<space>

- Entropy = expected amount of information contained in a random variable -> information is synonymous with "bits" which is why is log, base 2 
     - the more a feature splits the data in obvious ways, the less informative it is for us, entropy is lower
     - the more the feature splits the data, the higher the entropy and hence information gained by splitting at that feature
     - Entropy is minimized when one of the events has a P(X)=1
     - Entropy is maximized when each event has a P(X)=1/n of happening


----

## Trees
# Entropy calculation
<space>



----

## Trees
# Example
<space>


```r
voting_data <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')
str(voting_data)
names(voting_data) <- c('party','handicapped-infants','water-project-cost-sharing',
                        'adoption-of-the-budget-resolution','physician-fee-freeze',
                        'el-salvador-aid','religious-groups-in-schools',
                        'anti-satellite-test-ban','aid-to-nicaraguan-contras',
                        'mx-missile','immigration','synfuels-corporation-cutback',
                        'education-spending','superfund-right-to-sue','crime',
                        'duty-free-exports','export-administration-act-south-africa')


prop.table(table(voting_data[,1]))
n <- nrow(voting_data)
train_ind <- sample(n,2/3*n)
voting_train <- voting_data[train_ind,]
voting_test <- voting_data[-train_ind,]
```

----

## Trees
# Example
<space>

         party handicapped-infants water-project-cost-sharing
408   democrat                   y                          n
273 republican                   n                          n
406   democrat                   y                          n
138   democrat                   n                          n
271   democrat                   n                          y
426   democrat                   y                          n
    adoption-of-the-budget-resolution physician-fee-freeze el-salvador-aid
408                                 y                    n               n
273                                 n                    y               y
406                                 y                    n               y
138                                 y                    n               n
271                                 y                    n               n
426                                 y                    n               n
    religious-groups-in-schools anti-satellite-test-ban
408                           y                       y
273                           n                       y
406                           y                       n
138                           y                       y
271                           y                       y
426                           n                       y
    aid-to-nicaraguan-contras mx-missile immigration
408                         y          y           n
273                         y          n           y
406                         n          y           y
138                         y          y           y
271                         y          y           n
426                         y          y           y
    synfuels-corporation-cutback education-spending superfund-right-to-sue
408                            n                  y                      ?
273                            n                  y                      y
406                            n                  n                      y
138                            n                  n                      n
271                            ?                  n                      n
426                            n                  n                      n
    crime duty-free-exports export-administration-act-south-africa
408     y                 y                                      y
273     y                 ?                                      y
406     y                 n                                      y
138     y                 n                                      y
271     n                 n                                      y
426     n                 y                                      y
![plot of chunk tree_plot](figure/tree_plot.png) 

----

## Trees
# Example
<space>


```r
# make tree using C5.0
tree_model <- C5.0(voting_train[,-1],voting_train[,1], trials=1)
tree_predict <- predict(tree_model, newdata=voting_test[,-1])
conf <- CrossTable(voting_test[,1], tree_predict, prop.chisq = FALSE,
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
##     democrat |         89 |          6 |         95 | 
##              |      0.614 |      0.041 |            | 
## -------------|------------|------------|------------|
##   republican |          0 |         50 |         50 | 
##              |      0.000 |      0.345 |            | 
## -------------|------------|------------|------------|
## Column Total |         89 |         56 |        145 | 
## -------------|------------|------------|------------|
## 
## 
```

----

## Trees
# Example
<space>


```r
# most important variables
C5imp(tree_model)
```

```
##                                        Overall
## physician-fee-freeze                     96.89
## synfuels-corporation-cutback             42.21
## mx-missile                               10.73
## religious-groups-in-schools               2.08
## handicapped-infants                       0.00
## water-project-cost-sharing                0.00
## adoption-of-the-budget-resolution         0.00
## el-salvador-aid                           0.00
## anti-satellite-test-ban                   0.00
## aid-to-nicaraguan-contras                 0.00
## immigration                               0.00
## education-spending                        0.00
## superfund-right-to-sue                    0.00
## crime                                     0.00
## duty-free-exports                         0.00
## export-administration-act-south-africa    0.00
```

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
## C5.0 [Release 2.07 GPL Edition]  	Mon Aug 11 19:48:29 2014
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 289 cases (17 attributes) from undefined.data
## 
## Decision tree:
## 
## physician-fee-freeze in {?,n}: democrat (165.1/3.7)
## physician-fee-freeze = y:
## :...synfuels-corporation-cutback in {?,n}: republican (99.2/2.7)
##     synfuels-corporation-cutback = y:
##     :...mx-missile in {?,n}: republican (20/4.3)
##         mx-missile = y:
##         :...religious-groups-in-schools = n: republican (1)
##             religious-groups-in-schools in {?,y}: democrat (3.6)
## 
## 
## Evaluation on training data (289 cases):
## 
## 	    Decision Tree   
## 	  ----------------  
## 	  Size      Errors  
## 
## 	     5   10( 3.5%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	   167     5    (a): class democrat
## 	     5   112    (b): class republican
## 
## 
## 	Attribute usage:
## 
## 	 96.89%	physician-fee-freeze
## 	 42.21%	synfuels-corporation-cutback
## 	 10.73%	mx-missile
## 	  2.08%	religious-groups-in-schools
## 
## 
## Time: 0.0 secs
```

----

## Trees
# Example - Boosting
<space>

- rooted in the notion that by combining a number of weak performing learners, 
- you can create a team that is much stronger than any one of the learners alone.
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


 
   Cell Contents
|-------------------------|
|                       N |
|         N / Table Total |
|-------------------------|

 
Total Observations in Table:  145 

 
             | predicted class 
actual class |   democrat | republican |  Row Total | 
-------------|------------|------------|------------|
    democrat |         89 |          6 |         95 | 
             |      0.614 |      0.041 |            | 
-------------|------------|------------|------------|
  republican |          0 |         50 |         50 | 
             |      0.000 |      0.345 |            | 
-------------|------------|------------|------------|
Column Total |         89 |         56 |        145 | 
-------------|------------|------------|------------|

 

```r
# in-sample error rate
summary(boosted_tree_model)
```


Call:
C5.0.default(x = voting_train[, -1], y = voting_train[, 1], trials = 25)


C5.0 [Release 2.07 GPL Edition]  	Mon Aug 11 19:48:29 2014
-------------------------------

Class specified by attribute `outcome'

Read 289 cases (17 attributes) from undefined.data

-----  Trial 0:  -----

Decision tree:

physician-fee-freeze in {?,n}: democrat (165.1/3.7)
physician-fee-freeze = y:
:...synfuels-corporation-cutback in {?,n}: republican (99.2/2.7)
    synfuels-corporation-cutback = y:
    :...mx-missile in {?,n}: republican (20/4.3)
        mx-missile = y:
        :...religious-groups-in-schools = n: republican (1)
            religious-groups-in-schools in {?,y}: democrat (3.6)

-----  Trial 1:  -----

Decision tree:

adoption-of-the-budget-resolution in {?,y}: democrat (149.3/21.5)
adoption-of-the-budget-resolution = n:
:...synfuels-corporation-cutback in {?,n}: republican (100.7/11.7)
    synfuels-corporation-cutback = y: democrat (39/13.1)

-----  Trial 2:  -----

Decision tree:

physician-fee-freeze = n: democrat (128.5/23.2)
physician-fee-freeze in {?,y}: republican (160.5/37.7)

-----  Trial 3:  -----

Decision tree:

crime in {?,n}: democrat (77.4/9.4)
crime = y:
:...synfuels-corporation-cutback in {?,n}: republican (121.6/38.3)
    synfuels-corporation-cutback = y: democrat (90.1/33.1)

-----  Trial 4:  -----

Decision tree:

adoption-of-the-budget-resolution in {?,y}: democrat (139/29.9)
adoption-of-the-budget-resolution = n:
:...education-spending = n: democrat (56/19.9)
    education-spending in {?,y}: republican (94/16.5)

-----  Trial 5:  -----

Decision tree:

physician-fee-freeze in {?,n}: democrat (113.4/24)
physician-fee-freeze = y:
:...immigration = n: democrat (98/46.3)
    immigration in {?,y}: republican (77.6/5.2)

-----  Trial 6:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (100.5/24.9)
physician-fee-freeze = y:
:...immigration in {?,y}: republican (67.9/6.2)
    immigration = n:
    :...adoption-of-the-budget-resolution in {?,n}: republican (93.9/27.2)
        adoption-of-the-budget-resolution = y: democrat (26.7/9.6)

-----  Trial 7:  -----

Decision tree:

synfuels-corporation-cutback = ?: republican (0)
synfuels-corporation-cutback = n:
:...duty-free-exports in {?,n}: republican (115.7/24.2)
:   duty-free-exports = y: democrat (57.1/22.5)
synfuels-corporation-cutback = y:
:...physician-fee-freeze in {?,n}: democrat (28.5/1.5)
    physician-fee-freeze = y:
    :...mx-missile = n: republican (62.2/27.3)
        mx-missile in {?,y}: democrat (25.5/4.1)

-----  Trial 8:  -----

Decision tree:

physician-fee-freeze = ?: democrat (0)
physician-fee-freeze = n:
:...adoption-of-the-budget-resolution = n: republican (40/15.8)
:   adoption-of-the-budget-resolution in {?,y}: democrat (70.8/4.8)
physician-fee-freeze = y:
:...synfuels-corporation-cutback in {?,n}: republican (97.9/20.6)
    synfuels-corporation-cutback = y:
    :...immigration in {?,n}: democrat (56.3/17.2)
        immigration = y: republican (24/7.7)

-----  Trial 9:  -----

Decision tree:

physician-fee-freeze in {?,n}: democrat (118.1/26.5)
physician-fee-freeze = y:
:...superfund-right-to-sue in {?,n}: republican (21.4/2.4)
    superfund-right-to-sue = y:
    :...education-spending = ?: republican (0)
        education-spending = n: democrat (45.9/15.5)
        education-spending = y:
        :...anti-satellite-test-ban in {?,y}: republican (16.9/0.2)
            anti-satellite-test-ban = n:
            :...adoption-of-the-budget-resolution in {?,
                :                                     n}: republican (60.7/12.2)
                adoption-of-the-budget-resolution = y: democrat (26/6.3)

-----  Trial 10:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (109.7/33.3)
physician-fee-freeze = y:
:...synfuels-corporation-cutback in {?,n}: republican (96.8/19.4)
    synfuels-corporation-cutback = y:
    :...mx-missile = n: republican (60.6/25.5)
        mx-missile in {?,y}: democrat (21.9/4)

-----  Trial 11:  -----

Decision tree:

crime in {?,n}: democrat (40.2/8.8)
crime = y:
:...anti-satellite-test-ban in {?,y}: republican (101.1/26.2)
    anti-satellite-test-ban = n:
    :...physician-fee-freeze in {?,n}: democrat (15.2/0.9)
        physician-fee-freeze = y:
        :...immigration in {?,n}: democrat (109.2/39.4)
            immigration = y: republican (23.3/4.6)

-----  Trial 12:  -----

Decision tree:

adoption-of-the-budget-resolution in {?,y}: democrat (111.5/36.6)
adoption-of-the-budget-resolution = n:
:...synfuels-corporation-cutback in {?,n}: republican (102.3/24.6)
    synfuels-corporation-cutback = y: democrat (75.2/30.1)

-----  Trial 13:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (99/31.9)
physician-fee-freeze = y:
:...immigration in {?,y}: republican (61/8.4)
    immigration = n:
    :...duty-free-exports in {?,n}: republican (106.1/40.9)
        duty-free-exports = y: democrat (22.8/6.6)

-----  Trial 14:  -----

Decision tree:

adoption-of-the-budget-resolution = ?: republican (0)
adoption-of-the-budget-resolution = y: democrat (110.5/39.1)
adoption-of-the-budget-resolution = n:
:...synfuels-corporation-cutback in {?,n}: republican (94.1/25.9)
    synfuels-corporation-cutback = y: democrat (84.3/38.3)

-----  Trial 15:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (88.6/30.5)
physician-fee-freeze = y:
:...immigration in {?,y}: republican (67.7/13)
    immigration = n:
    :...anti-satellite-test-ban in {?,y}: republican (16.6/1.9)
        anti-satellite-test-ban = n:
        :...adoption-of-the-budget-resolution in {?,y}: democrat (23.4/1.7)
            adoption-of-the-budget-resolution = n:
            :...education-spending = n: democrat (39.7/14.9)
                education-spending in {?,y}: republican (53/16.3)

-----  Trial 16:  -----

Decision tree:

mx-missile = ?: republican (0)
mx-missile = y:
:...religious-groups-in-schools = n: republican (49.9/20.4)
:   religious-groups-in-schools in {?,y}: democrat (63.6/11.8)
mx-missile = n:
:...el-salvador-aid = ?: republican (0)
    el-salvador-aid = n: democrat (5/0.2)
    el-salvador-aid = y:
    :...anti-satellite-test-ban in {?,y}: republican (37.6/1.2)
        anti-satellite-test-ban = n:
        :...immigration in {?,y}: republican (24.6/2.1)
            immigration = n:
            :...adoption-of-the-budget-resolution in {?,
                :                                     y}: democrat (18.1/0.7)
                adoption-of-the-budget-resolution = n:
                :...physician-fee-freeze = n: democrat (3.8/0.1)
                    physician-fee-freeze in {?,y}: republican (86.5/35.2)

-----  Trial 17:  -----

Decision tree:

synfuels-corporation-cutback = ?: republican (0)
synfuels-corporation-cutback = n:
:...el-salvador-aid = n: democrat (74.7/26.2)
:   el-salvador-aid in {?,y}: republican (81.3/22.6)
synfuels-corporation-cutback = y:
:...physician-fee-freeze in {?,n}: democrat (35.6/1.1)
    physician-fee-freeze = y:
    :...superfund-right-to-sue = n: republican (7.1/0.9)
        superfund-right-to-sue in {?,y}: democrat (90.3/36.6)

-----  Trial 18:  -----

Decision tree:

adoption-of-the-budget-resolution in {?,n}: republican (170.5/65.8)
adoption-of-the-budget-resolution = y: democrat (118.5/34.2)

-----  Trial 19:  -----

Decision tree:

physician-fee-freeze in {?,n}: democrat (113.3/30)
physician-fee-freeze = y:
:...immigration in {?,y}: republican (59.7/10.8)
    immigration = n:
    :...export-administration-act-south-africa = n: democrat (37.4/12.8)
        export-administration-act-south-africa in {?,y}: republican (78.6/32.2)

-----  Trial 20:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (109.8/34.7)
physician-fee-freeze = y:
:...synfuels-corporation-cutback in {?,n}: republican (78.6/11.6)
    synfuels-corporation-cutback = y:
    :...mx-missile = n: republican (74.4/30.2)
        mx-missile in {?,y}: democrat (25.2/5.4)

-----  Trial 21:  -----

Decision tree:

synfuels-corporation-cutback = ?: republican (0)
synfuels-corporation-cutback = n:
:...adoption-of-the-budget-resolution in {?,n}: republican (85.9/13.3)
:   adoption-of-the-budget-resolution = y: democrat (64.2/25.4)
synfuels-corporation-cutback = y:
:...physician-fee-freeze in {?,n}: democrat (25.9/1.1)
    physician-fee-freeze = y:
    :...superfund-right-to-sue = ?: democrat (0)
        superfund-right-to-sue = n: republican (12.3/1.2)
        superfund-right-to-sue = y:
        :...mx-missile in {?,y}: democrat (17.6/0.7)
            mx-missile = n:
            :...immigration in {?,n}: democrat (70.2/23.9)
                immigration = y: republican (11.9)

-----  Trial 22:  -----

Decision tree:

crime = ?: republican (0)
crime = n: democrat (45.5/9.5)
crime = y:
:...superfund-right-to-sue in {?,n}: republican (66.7/13.7)
    superfund-right-to-sue = y:
    :...physician-fee-freeze = ?: republican (0)
        physician-fee-freeze = n: democrat (19.2/0.7)
        physician-fee-freeze = y:
        :...duty-free-exports in {?,y}: republican (12.8/0.1)
            duty-free-exports = n:
            :...el-salvador-aid = ?: republican (0)
                el-salvador-aid = n: democrat (7.3/2)
                el-salvador-aid = y:
                :...anti-satellite-test-ban in {?,y}: republican (21.4/0.2)
                    anti-satellite-test-ban = n: [S1]

SubTree [S1]

adoption-of-the-budget-resolution in {?,n}: republican (83.9/29.1)
adoption-of-the-budget-resolution = y: democrat (31.1/5.4)

-----  Trial 23:  -----

Decision tree:

physician-fee-freeze = ?: republican (0)
physician-fee-freeze = n: democrat (127.3/32)
physician-fee-freeze = y:
:...water-project-cost-sharing in {?,n}: republican (39.6/1.3)
    water-project-cost-sharing = y:
    :...immigration in {?,y}: republican (36.8/6.4)
        immigration = n:
        :...adoption-of-the-budget-resolution in {?,n}: republican (61.3/20.1)
            adoption-of-the-budget-resolution = y: democrat (22/1.6)

-----  Trial 24:  -----

Decision tree:

mx-missile in {?,y}: democrat (135.8/22.4)
mx-missile = n:
:...handicapped-infants in {?,n}: republican (88.9/16.1)
    handicapped-infants = y: democrat (60.3/18.7)


Evaluation on training data (289 cases):

Trial	    Decision Tree   
-----	  ----------------  
	  Size      Errors  

   0	     5   10( 3.5%)
   1	     3   34(11.8%)
   2	     2   16( 5.5%)
   3	     3   42(14.5%)
   4	     3   29(10.0%)
   5	     3   50(17.3%)
   6	     4   13( 4.5%)
   7	     5   41(14.2%)
   8	     5   28( 9.7%)
   9	     6   16( 5.5%)
  10	     4    8( 2.8%)
  11	     5   73(25.3%)
  12	     3   34(11.8%)
  13	     4   14( 4.8%)
  14	     3   34(11.8%)
  15	     6   12( 4.2%)
  16	     8   96(33.2%)
  17	     5   31(10.7%)
  18	     2   35(12.1%)
  19	     4   24( 8.3%)
  20	     4    8( 2.8%)
  21	     7   23( 8.0%)
  22	     8   38(13.1%)
  23	     5   11( 3.8%)
  24	     3   50(17.3%)
boost	          6( 2.1%)   <<


	   (a)   (b)    <-classified as
	  ----  ----
	   170     2    (a): class democrat
	     4   113    (b): class republican


	Attribute usage:

	 96.89%	physician-fee-freeze
	 96.54%	adoption-of-the-budget-resolution
	 96.19%	synfuels-corporation-cutback
	 95.85%	mx-missile
	 95.50%	crime
	 78.89%	el-salvador-aid
	 65.05%	duty-free-exports
	 63.32%	anti-satellite-test-ban
	 59.86%	superfund-right-to-sue
	 51.90%	handicapped-infants
	 49.83%	religious-groups-in-schools
	 49.13%	immigration
	 45.33%	education-spending
	 38.41%	water-project-cost-sharing
	 14.88%	export-administration-act-south-africa


Time: 0.0 secs

----

## Trees
# Example - Error Cost
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
##     democrat |         87 |          8 |         95 | 
##              |      0.600 |      0.055 |            | 
## -------------|------------|------------|------------|
##   republican |          0 |         50 |         50 | 
##              |      0.000 |      0.345 |            | 
## -------------|------------|------------|------------|
## Column Total |         87 |         58 |        145 | 
## -------------|------------|------------|------------|
## 
## 
```

----

## Trees
# Example - Error Cost
<space>


```r
tris <- seq(1,50,by=2)
boost_acc <- NULL
for (i in tris){  
  temp <- C5.0(voting_train[,-1],voting_train[,1], trials=i, costs = error_cost)
  temp_pred <- predict(temp,voting_test[,-1])
  boost_acc <- append(boost_acc,sum(diag(table(temp_pred,voting_test[,1]))))
}
```

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

```r
plot(boost_acc,type='l')
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10.png) 

----

## Trees
# Pros and Cons
<space>

- trees are non-parametric, rule based classification or regression method
- simple to understand and interpret
- little data preparation
- easy to overfit (need to prune to avoid that, or have max tree depth)
- usually finds local optimum. Can mitigate this with an ensemble of trees
- difficult concepts that are not easily expressed by trees (XOR) are hard to learn
- for class imbalance, trees can be biased - should balance dataset before fitting
- trees tend to overfit, so use PCA beforehand

----

## Missing Data
# Types
<space>

Missingness that...
- is completely at random; no bias in missing data
- is random
- depends on unobserved features
- depends on the missing value itself
http://www.stat.columbia.edu/~gelman/arm/missing.pdf
----
## Resources
<space>

- [Machine Learning with R](http://www.packtpub.com/machine-learning-with-r/book)
- [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
- [Elements of Statistical Learning](http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)

----
