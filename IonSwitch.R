start_time <- Sys.time()
suppressMessages(library(randomForest))
suppressMessages(library(glmnet))
suppressMessages(library(tidyverse))
suppressMessages(library(caret))
suppressMessages(library(dplyr))
suppressMessages(library(rpart))
library(e1071) 
#here we are reading the data to train
train<-read.csv("train (1).csv")
#here we are reading the data to test
test<-read.csv("test (1).csv")
test<-test[,c(1,2)]

#we are taking the samples fomr the total training data.
x<-train[sample(nrow(train))[c(1:200000)],c(2,3)]
x_train<-x[sample(0.8*nrow(x)),c(1,2)]
x_train[,2]<-as.factor(x_train[,2])
x_val<-x[-sample(0.8*nrow(x)),c(1,2)]
x_val[,2]<-as.factor(x_val[,2])
t<-as.data.frame(x_val[,-2])
colnames(t)<-'signal'

#we are building the model using the random forest
model<-randomForest(open_channels~.,data =x_train,importance = TRUE)
pred<-predict(model,newdata=t,type="response")

#creating the confusion matrix and storing it in the result
result<-confusionMatrix(pred,x_val[,2])[4]
mean(result$byClass[,11])#val_accuracy:
#rm(list=ls())
preds = data.frame(c(11:2000010))
n= 40

#we are training the data with 40 samples.
for (i in 1:n){ 
  
  x<-train[sample(nrow(train))[c(1:100000)],c(2,3)]
  x_train<-x[sample(0.8*nrow(x)),c(1,2)]
  x_train[,2]<-as.factor(x_train[,2])
  model<-randomForest(open_channels~.,data =x_train,importance = TRUE)
  pred<-predict(model,newdata=subset(test),type="response")
  preds <-cbind(preds,pred)
  
}

colnames(preds)<-c(0:n)
preds<-preds[,c(2:n)]
head(preds)

#creating a CSV from the prediction data.
write.csv(preds,'Group_7_submission.csv',row.names = FALSE,quote = FALSE)
test<-test[,1]
y_final = read.csv("Group_7_submission.csv")
y_final<-apply(y_final, 1,median)
df<-data.frame(cbind(test,y_final))
colnames(df)<-c("time","open_channels")
write.csv(df,'Group_7_submission.csv',row.names = FALSE,quote = FALSE)
#hist(df$open_channels)
end_time <- Sys.time()
end_time - start_time
rm(list=ls())