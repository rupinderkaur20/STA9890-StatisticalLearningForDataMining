## STA9890 Project
## Group 18, Members: Rupinder Kaur and Janani Ravichandran
## Cryptocurrency data set


rm(list = ls()) #delete objects in the global Environment
cat("\014")

library(Matrix)
library(glmnet)
library(randomForest)
library(reshape2)
library(tidyverse)
library(ggpubr)

## reading the file
crypt = read.csv("Crypto.csv")

# Cleaning the data
## Removing some lagged variables and columns that contains id's
exclude.vars = c("id", "asset_id", "market_cap_rank", "high", "low", 
          "market_cap_global", "market_cap", "open" )
crypt = select(crypt, -exclude.vars)

## missing values
sum(is.na(crypt)) # there are 13025 missing values

## removing the rows with missing values
crypt = na.omit(crypt) # 2391 rows left

# predictors and response variable
X = as.matrix(crypt[, 1:40])
Y = as.matrix(crypt[, 41])

n = dim(X)[1]
p =  dim(X)[2]

#summary statistics of the response variable
summary(crypt$close)
hist(crypt$close)
sd(crypt$close)

# size of the loop
k = 10

# empty Data frame to store R-squares
rsq.df = data.frame(matrix(0, nrow = k * 4, ncol = 4))

for (K in c(1:k)) {
  cat("Loop", K, "\n")
  
  ## splitting data into training and test sets
  i.mix = sample(n) #shuffling
  train = i.mix[1:floor(n*0.8)]
  
  #Training predictors and response
  X.train = X[train,]
  Y.train = Y[train]
  
  # Validation predictors and response
  X.test = X[-train,]
  Y.test = Y[-train]
  
  ## 10 fold cross validation and recording the time it takes
  #Lasso
  lasso.cv.start = Sys.time()
  cv.lasso = cv.glmnet(X.train, Y.train, alpha = 1)
  lasso.cv.time = round(Sys.time() - lasso.cv.start, 1)
  
  #elastic net
  elnet.cv.start = Sys.time()
  cv.elnet = cv.glmnet(X.train, Y.train, alpha = 0.5)
  elnet.cv.time = round(Sys.time() - elnet.cv.start, 1)
  
  #ridge regression
  ridge.cv.start = Sys.time()
  cv.ridge = cv.glmnet(X.train, Y.train, alpha = 0)
  ridge.cv.time = round(Sys.time() - ridge.cv.start , 1)
 
  ## Fit Lasso, Ridge , elastic net and random forest
  #Lasso
  lasso.fit = glmnet(X.train, Y.train, axis= 1, lambda = cv.lasso$lambda.min)
  
  #Elastic net
  elnet.fit = glmnet(X.train, Y.train, axis= 0.5, lambda = cv.elnet$lambda.min)
  
  #Ridge regression
  ridge.fit = glmnet(X.train, Y.train, axis= 0, lambda = cv.ridge$lambda.min)
  
  ##Random Forest
  rf.fit = randomForest(x = X.train, y= Y.train, mtry = p/5, 
                            importance = TRUE )
  
  ## Calculating Training R-Square
  # Lasso
  yhat.lasso = predict(lasso.fit, X.train)
  train.rsq.lasso = cor(Y.train, yhat.lasso)^2
  
  # elastic net
  yhat.elnet = predict(elnet.fit, X.train)
  train.rsq.elnet = cor(Y.train, yhat.elnet)^2
  
  # Ridge Regression
  yhat.ridge = predict(ridge.fit, X.train)
  train.rsq.ridge = cor(Y.train, yhat.ridge)^2
  
  #Random Forest
  yhat.rf = predict(rf.fit, X.train)
  train.rsq.rf = cor(Y.train, yhat.rf)^2
  
  ## Calculating test R-square
  
  # total sum of squares
  tss = mean((Y.test - mean(Y.test))^2)
  
  # Lasso
  yhat.test.lasso = predict(lasso.fit, X.test)
  lasso.rss = mean((Y.test- yhat.test.lasso)^2) # Residual Sum of Squares
  test.rsq.lasso = 1-(lasso.rss / tss)
  
  # Elastic net
  yhat.test.elnet = predict(elnet.fit, X.test)
  elnet.rss = mean((Y.test- yhat.test.elnet)^2) # Residual Sum of Squares
  test.rsq.elnet = 1-(elnet.rss / tss)
  
  #Ridge Regression
  yhat.test.ridge = predict(ridge.fit, X.test)
  ridge.rss = mean((Y.test- yhat.test.ridge)^2) # Residual Sum of Squares
  test.rsq.ridge = 1-(ridge.rss / tss)
  
  # Random Forest
  yhat.test.rf = predict(rf.fit, X.test)
  rf.rss = mean((Y.test- yhat.test.rf)^2) # Residual Sum of Squares
  test.rsq.rf = 1-(rf.rss / tss)
  
  ## updating the R-square Data frame 
  rsq.df[(K*4-3):(K*4),] = rbind(cbind(K, "Lasso", train.rsq.lasso, test.rsq.lasso), 
                                 cbind(K, "Elastic Net", train.rsq.elnet, test.rsq.elnet), 
                                 cbind(K, "Ridge Regression", train.rsq.ridge, test.rsq.ridge),
                                 cbind(K, "Random Forest", train.rsq.rf, test.rsq.rf))
  
 }# end of loop

## Box plots of training and test R-square

# changing the names of the column
names(rsq.df) = c("loop", "model", "train_rsq", "test_rsq")

# change the training and test R-square values to numeric
rsq.df[,c("train_rsq", "test_rsq")] = lapply(rsq.df[,c("train_rsq", "test_rsq")], as.numeric)

# Changing the "model" column from character to factor data type
rsq.df$model = as.factor(rsq.df$model)


p1 <- ggplot(data = rsq.df, mapping = aes(x = model, y = train_rsq, fill = model))+
  geom_boxplot() + labs(x = "Models", y = "Training R- Square") + 
  scale_x_discrete(labels = c("Elastic Net", "Lasso", "R.Forest", "Ridge")) +
  theme(legend.position = "none")

p2 <- ggplot(data = rsq.df, mapping = aes(x = model, y = test_rsq, fill = model))+
  geom_boxplot() + labs(x = "Models", y = "Testing R- Square") +
  scale_x_discrete(labels = c("Elastic Net", "Lasso", "R.Forest", "Ridge")) +
  theme(legend.position = "none")

ggarrange(p1, p2, widths = c(2, 2))


#Cross validation curves for the 100th sample
plot(cv.lasso)
title(paste("Lasso, ", "CV Time: ", lasso.cv.time, units(lasso.cv.time)), line= 2.5)
plot(cv.elnet)
title(paste("Elastic Net, ", "CV Time: ", elnet.cv.time, units(elnet.cv.time)),line = 2.5)
plot(cv.ridge)
title(paste("Ridge, ", "CV Time: ", ridge.cv.time, units(ridge.cv.time)), line =2.5)

## box plot for test and train residuals
# Data frame for residuals of the training data
res.train = data.frame(rbind(cbind("Lasso", Y.train - yhat.lasso),
                             cbind("Elastic Net", Y.train - yhat.elnet),
                             cbind("Ridge Regression", Y.train - yhat.ridge),
                             cbind("Random Forest", Y.train - yhat.rf)))

names(res.train) = c("model", "residual")
res.train$model = as.factor(res.train$model)
res.train$residual = as.numeric(res.train$residual)

# Data frame for residuals of the test data
res.test = data.frame(rbind(cbind("Lasso", Y.test - yhat.test.lasso),
                            cbind("Elastic Net", Y.test - yhat.test.elnet),
                            cbind("Ridge Regression", Y.test - yhat.test.ridge),
                            cbind("Random Forest", Y.test - yhat.test.rf)))

names(res.test) = c("model", "residual")
res.test$model = as.factor(res.test$model)
res.test$residual = as.numeric(res.test$residual)

# box plot: train residuals
r1 <- ggplot(data = res.train, mapping = aes(x = model, y= residual, fill = model)) + 
  geom_boxplot() + labs(x= "Model", y = "Train Residual") + 
  scale_x_discrete(labels = c("Elastic Net", "Lasso", "R.Forest", "Ridge")) +
  theme(legend.position = "none")

r2 <- ggplot(data = res.test, mapping = aes(x = model, y= residual, fill = model)) + 
  geom_boxplot() + labs(x= "Model", y = "Test Residual") +
  scale_x_discrete(labels = c("Elastic Net", "Lasso", "R.Forest", "Ridge")) +
  theme(legend.position = "none")

ggarrange(r1, r2, widths = c(2, 2))

## fitting all four models on all of the data
# Lasso
lasso.full.start = Sys.time() 
lasso.cv.full = cv.glmnet(x = X, y= Y, axis = 1)
lasso.fit.full = glmnet(x = X, y= Y, axis = 1, lambda = lasso.cv.full$lambda.min)
lasso.full.time = round(Sys.time() - lasso.full.start,1)

#elastic net
elnet.full.start = Sys.time() 
elnet.cv.full = cv.glmnet(x = X, y= Y, axis = 0.5)
elnet.fit.full = glmnet(x = X, y= Y, axis = 0.5, lambda = elnet.cv.full$lambda.min)
elnet.full.time = round(Sys.time() - elnet.full.start,1)

# Ridge Regression
ridge.full.start = Sys.time() 
ridge.cv.full = cv.glmnet(x = X, y= Y, axis = 0)
ridge.fit.full = glmnet(x = X, y= Y, axis = 0, lambda = ridge.cv.full$lambda.min)
ridge.full.time = round(Sys.time() - ridge.full.start,1)

# Random Forest
rf.full.start = Sys.time() 
rf.fit.full = randomForest(x = X, y= Y, mtry = p/5, importance = TRUE )
rf.full.time = round(Sys.time() - rf.full.start,1)

# 90% test interval for the R-squared values
lasso.interval = quantile(rsq.df$test_rsq[rsq.df[,"model"] == "Lasso"], probs=c(0.05, 0.95))
lasso.interval = paste(round(lasso.interval[1],3), round(lasso.interval[2],3), sep="-")

elnet.interval =  quantile(rsq.df$test_rsq[rsq.df[,"model"] == "Elastic Net"], probs=c(0.05, 0.95))
elnet.interval = paste(round(elnet.interval[1],3), round(elnet.interval[2],3), sep="-")

ridge.interval = quantile(rsq.df$test_rsq[rsq.df[,"model"] == "Ridge Regression"], probs=c(0.05, 0.95))
ridge.interval = paste(round(ridge.interval[1],3), round(ridge.interval[2],3), sep="-")

rf.interval = quantile(rsq.df$test_rsq[rsq.df[,"model"] == "Random Forest"], probs=c(0.05, 0.95))
rf.interval = paste(round(rf.interval[1],3), round(rf.interval[2],3), sep="-")

interval.df = data.frame(interval = c(ridge.interval, lasso.interval, elnet.interval, rf.interval), 
                         time = c(ridge.full.time, lasso.full.time, elnet.full.time, rf.full.time))

##variable importance
# creating a data frame for coefficients and importance
beta.df = data.frame(Variable = seq(1,p,by=1),
                     Lasso = as.numeric(lasso.fit.full$beta), 
                     Elastic.Net = as.numeric(elnet.fit.full$beta),
                     Ridge = as.numeric(ridge.fit.full$beta),
                     Random.Forest = rf.fit.full$importance[,1])

#ordering the coefficients using the elastic net as reference
beta.df = beta.df %>% 
  arrange(Elastic.Net) %>% 
  mutate(Order = seq(p,1,by=-1))

write.csv(beta.df, "VariableImportance.csv")

# Code the variable names 
beta.df$Variable = paste("V",beta.df$Variable, sep = "")
beta.df$Variable = factor(beta.df$Variable)

# Sort the data from high to low
beta.df$Variable = reorder(beta.df$Variable, beta.df$Order)

# Reshape the data frame
beta.df = melt(beta.df, id.vars = c("Variable", "Order"))

# Rename the columns name 
beta.df = beta.df %>% 
  rename(Coefficient = value, Model = variable) %>% 
  mutate(pos = Coefficient >= 0)

# Plot the coefficients and importance
ggplot(beta.df) + 
  aes(x=Variable, y=Coefficient, fill = pos) + 
  geom_col(color = "black", show.legend = FALSE) + 
  facet_grid(Model ~., scales="free") + 
  theme_linedraw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
  theme(axis.text.x = element_text(size = 6, angle = 90, hjust = 1)) + 
  theme(axis.text.y = element_text(size = 6)) + 
  ylab("Coefficient Value") 

