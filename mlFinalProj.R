library(caret)
library(ggplot2)
library(Metrics)

train <- "pml-training.csv"
test <- "pml-testing.csv"

train.Data <- read.csv(train, na.strings=c("NA","#DIV/0!",""))
test.Data <- read.csv(test, na.strings=c("NA","#DIV/0!",""))

## Removing Erroneous NaN
NAindex <- apply(train.Data,2,function(x) {sum(is.na(x))}) 
train.Data <- train.Data[,which(NAindex == 0)]
NAindex <- apply(test.Data,2,function(x) {sum(is.na(x))}) 
test.Data <- test.Data[,which(NAindex == 0)]

## Preprocessing Data
v <- which(lapply(train.Data, class) %in% "numeric")

preObj <-preProcess(train.Data[,v],method=c('knnImpute', 'center', 'scale'))
trainLoesser <- predict(preObj, train.Data[,v])
trainLoesser$classe <- train.Data$classe

testLoesser <-predict(preObj, test.Data[,v])
testLoesser$classe <- test.Data$classe

## Removing Near Zero Values
nzv <- nearZeroVar(trainLoesser,saveMetrics=TRUE)
trainLoesser <- trainLoesser[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testLoesser,saveMetrics=TRUE)
testLoesser <- testLoesser[,nzv$nzv==FALSE]

## Creating Data Partition
set.seed(3888383)

trainIndex <- createDataPartition(y = trainLoesser$classe, p = .75, list = FALSE)
training = trainLoesser[trainIndex, ]
testing = trainLoesser[-trainIndex, ]

control = trainControl(method = 'cv', number = 4)

## Fitting the model 

randomForestMod <- train(classe ~.,
                         data = training,
                         method = "rf",
                         trControl = control)

randomForestMod
summary(randomForestMod$finalModel)

## Evaluating the model
pred <- predict(randomForestMod, newdata = testing); 

confusionMatrix(pred, testing$classe)

