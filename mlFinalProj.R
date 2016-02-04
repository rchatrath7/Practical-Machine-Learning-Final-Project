---
title: "Practical Machine Learning Final Project"
author: "Rakesh Chatrath"
output: html_document
---

```{r}
library(caret)
library(ggplot2)
library(Metrics)

trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train <- "training.csv"
test <- "testing.csv"

download.file(trainURL, train, method = "curl")
download.file(testURL, test, method = "curl")

train.Data <- read.csv(train, na.strings=c("NA","#DIV/0!",""))
test.Data <- read.csv(test, na.strings=c("NA","#DIV/0!",""))
```

## Removing Erroneous NaN 
The following scripting was used to remove the over abundance of NaN values in columns. This was to ensure that the the train had reproducibility and was accurate. 
```{r}
NAindex <- apply(train.Data,2,function(x) {sum(is.na(x))}) 
train.Data <- train.Data[,which(NAindex == 0)]
NAindex <- apply(test.Data,2,function(x) {sum(is.na(x))}) 
test.Data <- test.Data[,which(NAindex == 0)]
```

## Preprocessing Data
After removing the erroneous NA's, the data needs to be preprocessed. Imputation was done so that the data set was tidy enough to train on and esnure consistent, accurate results. 
```{r}
v <- which(lapply(train.Data, class) %in% "numeric")

preObj <-preProcess(train.Data[,v],method=c('knnImpute', 'center', 'scale'))
trainLoesser <- predict(preObj, train.Data[,v])
trainLoesser$classe <- train.Data$classe

testLoesser <-predict(preObj, test.Data[,v])
testLoesser$classe <- test.Data$classe
```

## Removing Near Zero Values
There were a large number of near zero values when inspecting the data. The nature of these near zero values introduce unwanted variance when training on the data set. Thus, it was decided to remove them. 

```{r}
nzv <- nearZeroVar(trainLoesser,saveMetrics=TRUE)
trainLoesser <- trainLoesser[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testLoesser,saveMetrics=TRUE)
testLoesser <- testLoesser[,nzv$nzv==FALSE]
```

## Creating Data Partition
The traininga data was partitioned into 75% and 25% training/testing cross validated sets. 

```{r}
set.seed(3888383)

trainIndex <- createDataPartition(y = trainLoesser$classe, p = .75, list = FALSE)
training = trainLoesser[trainIndex, ]
testing = trainLoesser[-trainIndex, ]

control = trainControl(method = 'cv', number = 4)
```

## Fitting the model 
Random Forests were used with the caret packages `train()` function to fit a model on the data set. 
```{r eval = False}
randomForestMod <- train(classe ~.,
                         data = training,
                         method = "rf",
                         trControl = control,
                         prox = TRUE,
                         allowParallel = TRUE)
```

```{r}
randomForestMod
summary(randomForestMod$finalModel)
```

## Evaluating the model
From the confusion matrix, we can see that the accuracy of the model is extremely high, almost near perfect. 
```{r}
pred <- predict(randomForestMod, newdata = testing); 

confusionMatrix(pred, testing$classe)
```

