---

title       : Machine Learning with R - Part 2
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


----

## Logistic Regression
# Motivation
<space>


```r
library("MASS")
data(menarche)
log_data <- data.frame(Y=menarche$Menarche/menarche$Total)
log_data$X <- menarche$Age

glm.out < glm(cbind(Menarche, Total-Menarche) ~ Age,family=binomial(logit), data=menarche)
```

```
## Error: object 'glm.out' not found
```

```r
lm.out <- lm(Y~X, data=log_data)

log_data$fitted <- glm.out$fitted
```

```
## Error: object 'glm.out' not found
```

```r
data_points <- ggplot(log_data) + geom_point(aes(x=X,y=Y),color='blue',size=3)
```

```
## Error: could not find function "ggplot"
```

```r
line_points <- data_points + geom_abline(intercept = coef(lm.out)[1], slope = coef(lm.out)[2],color='green',size=1)
```

```
## Error: object 'data_points' not found
```

```r
curve_points <- line_points + geom_line(aes(x=X,y=fitted),color='red',size=1) 
```

```
## Error: object 'line_points' not found
```

----

## Logistic Regression
# Notation
<space>

- introduce notation: hypothesis function, cost function, objective function, sigmoid

----

## Logistic Regression
# Motivation
<space>

# logistic function - odds ratio - log odds

----

## Regression example
# Gradient descent
<space>


```
## Error: could not find function "ggplot"
```

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
[1] 5236
```

```r
iterations = 10000
alpha = 0.1
results <- gradDescent(X1,Y1,init_theta,iterations,alpha)
```

----

## Regression example
# Gradient descent
<space>


```
## Error: could not find function "ggplot"
```

----

## Regression example
# Gradient descent
<space>


```r
grad_cost(X1,Y1,theta[[1]])
```

```
[1] 338.1
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

```
## Error: could not find function "ggplot"
```

```r
ggplot(data=df,aes(x=X,y=Y))+geom_point()+geom_point(data=new_preds,aes(x=X,y=Y,color='red'),size=3)+scale_colour_discrete(guide = FALSE)
```

```
## Error: could not find function "ggplot"
```

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
# Motivation
<space>

# example

----

## Logistic Regression
# Motivation
<space>

# Summary

----

## Principle Component Analysis
# Motivation
<space>

# motivation for PCA

----

## Principle Component Analysis
# Motivation
<space>

# brief overview of important linear algebra theorems

----

## Principle Component Analysis
# Motivation
<space>

# where L is our eigenvalues and x is eigenvectors and A is our square matrix
# Ax = Lx
# Ax - LIx = 0
# (A-LI)x = 0
# what is x such that x is not all zero?
# determinant of A - LI must be 0
# the solution 

----

## Principle Component Analysis
# Motivation
<space>

A = matrix(c(5,2,2,5),nrow=2)
|A - L*diag(nrow(A))| = 0
det(c(5-l,2,2,5-l))
(5-l)*(5-l) - 4 = 0
25 - 10l + l^2 - 4 = 0
l^2 - 10l + 21 = 0
roots <- Re(polyroot(c(21,-10,1)))

----

## Principle Component Analysis
# Motivation
<space>

# when lambda = -3
Ax = 3x
5x1 + 2x2 = 3x1
2x1 + 5x2 = 3x2
x1=-x2
# one eigenvector = [1 -1]. any scalar multiple of this counts.

----

## Principle Component Analysis
# Motivation
<space>

# when lambda = 7
5x1 + 2x2 = 7x1
2x2 + 5x2 = 7x2
x1 = x2
# another eigenvector = [1 1]. any scalar multiple of this counts.

----

## Principle Component Analysis
# Motivation
<space>

A%*%c(1,-1) == 3 * as.matrix(c(1,-1))
A%*%c(1,1) == 7 * as.matrix(c(1,1))
roots

----

## Principle Component Analysis
# Motivation
<space>

- check
m <- matrix(c(1,-1,1,1),ncol=2)
m <- m/sqrt(norm(m))
A == as.matrix(m%*%diag(roots)%*%t(m))
# lambda is a diagonal matrix, with 0 off diagonals

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

data <- read.csv('tennis_data_2013.csv')
data$Player1 <- as.character(data$Player1)
data$Player2 <- as.character(data$Player2)

tennis <- data
m <- length(data)

for (i in 10:m){
  tennis[,i] <- ifelse(is.na(data[,i]),0,data[,i])
}

str(tennis)

features <- tennis[,10:m]

head(features)
str(features)
dim(features)

----

## Principle Component Analysis
# Example
<space>

scaled_features <- as.matrix(scale(features))
Cx <- cov(scaled_features)
eigenvalues <- eigen(Cx)$values
eigenvectors <- eigen(Cx)$vectors
PC <- scaled_features %*% eigenvectors

----

## Principle Component Analysis
# Example
<space>

Cy <- cov(PC)
sum(round(diag(Cy) - eigenvalues,5))
sum(round(Cy[upper.tri(Cy)],5)) ## off diagonals are 0 since PC's are orthogonal

----

## Principle Component Analysis
# Example
<space>

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

----

## Principle Component Analysis
# Example
<space>

eigenvalues = (pca.df$sdev)^2
eigenvectors are the loadings - linear combination of the variables
eigenvectors[,1] = pca.df$rotation[,1]
sum((eigenvectors[,1])^2)

----

## Principle Component Analysis
# Example
<space>

pca.df <- prcomp(scaled_features)
rows <- nrow(tennis)
pca.plot <- as.data.frame(pca.df$x[,1:2])
pca.plot$gender <- data$Gender
ggplot(data=pca.plot,aes(x=PC1,y=PC2,color=gender)) + geom_point()

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

- what is clustering

----

## Clustering
# Kmeans
<space>

- k means algorithm

----

## Clustering
# Kmeans
<space>

- k means algorithm

----

## Clustering
# Kmeans
<space>

- k means algorithm

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

x <- c(2,2,8,5,7,6,1,4)
y <- c(10,5,4,8,5,4,2,9)
cluster <- data.frame(X=c(x,2*x,3*x),Y=c(y,-2*x,1/4*y))
plot(cluster)

----

## Clustering
# DBSCAN
<space>

install.packages('fpc')
library(fpc)
cluster_DBSCAN<-dbscan(cluster, eps=3, MinPts=2, method="hybrid")
plot(cluster_DBSCAN, cluster, main="Clustering using DBSCAN algorithm (eps=3, MinPts=3)")

----

## Clustering
# Summary
<space>

----

## Trees
# Motivation
<space>

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

- what is entropy?

----

## Trees
# Entropy calculation
<space>

entropy_function <- function(p)
  {
    if (min(p) < 0 || sum(p) <= 0)
      return(NA)
    p.norm <- p[p>0]/sum(p)
    -sum(log2(p.norm)*p.norm)
  }

entropy_function(c(0.99,1))

----

## Trees
# Example
<space>

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

----

## Trees
# Example
<space>

install.packages("party")
library(C50)
library(party)
library(gmodels)
head(voting_train)

# plot tree using party package
tree_formula <- with(voting_train,voting_train$party ~ .)
p_tree <- ctree(tree_formula,data=voting_train)
plot(p_tree,
     inner_panel=node_inner(p_tree,pval = FALSE,id = TRUE),
     terminal_panel=node_terminal(p_tree, digits = 1, id = TRUE))

----

## Trees
# Example
<space>

# make tree using C5.0
tree_model <- C5.0(voting_train[,-1],voting_train[,1], trials=1)
tree_predict <- predict(tree_model, newdata=voting_test[,-1])
conf <- CrossTable(voting_test[,1], tree_predict, prop.chisq = FALSE,
                   prop.c = FALSE, prop.r = FALSE,
                   dnn = c("actual class", "predicted class"))

----

## Trees
# Example
<space>

# most important variables
C5imp(tree_model)

# in-sample error rate
summary(tree_model)

----

## Trees
# Example - Boosting
<space>

# boosting is rooted in the notion that by combining a number of weak performing learners, 
# you can create a team that is much stronger than any one of the learners alone.
# this is where C5.0 improves on C4.5

----

## Trees
# Example - Boosting
<space>

boosted_tree_model <- C5.0(voting_train[,-1],voting_train[,1], trials=25)
boosted_tennis_predict <- predict(boosted_tree_model,voting_test[,-1])

boosted_conf <- CrossTable(voting_test[,1], boosted_tennis_predict, prop.chisq = FALSE,
                           prop.c = FALSE, prop.r = FALSE, 
                           dnn = c("actual class", "predicted class"))

# in-sample error rate
summary(boosted_tree_model)

----

## Trees
# Example - Error Cost
<space>

# still getting too many false positives (predict republican but actually democrat)
# introduce higher cost to getting this wrong

error_cost <- matrix(c(0,1,2,0),nrow=2)
cost_model <- C5.0(voting_train[,-1],voting_train[,1], trials=1, costs = error_cost)
cost_predict <- predict(cost_model, newdata=voting_test[,-1])
conf <- CrossTable(voting_test[,1], cost_predict, prop.chisq = FALSE,
                   prop.c = FALSE, prop.r = FALSE,
                   dnn = c("actual class", "predicted class"))

----

## Trees
# Example - Error Cost
<space>

tris <- seq(1,50,by=2)
boost_acc <- NULL
for (i in tris){  
  temp <- C5.0(voting_train[,-1],voting_train[,1], trials=i, costs = error_cost)
  temp_pred <- predict(temp,voting_test[,-1])
  boost_acc <- append(boost_acc,sum(diag(table(temp_pred,voting_test[,1]))))
}

plot(boost_acc,type='l')

----

## Trees
# Pros and Cons
<space>

----

## Resources
<space>

- [Machine Learning with R](http://www.packtpub.com/machine-learning-with-r/book)
- [Machine Learning for Hackers](http://shop.oreilly.com/product/0636920018483.do)
- [Elements of Statistical Learning](http://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)

----
