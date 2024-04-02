library(randomForest)
setwd("D:/") 
for(d in 1:3)
{
  a=paste('/seven/new/',d,'d')
  a=gsub(" ",'',a,fixed=TRUE)
  fileC <- list.files(a)             
  dirC <- paste(".",a,'/',fileC,sep="") 
  n<-length(dirC)
  i=1
  for(i in 1:59)
  {
    data<-read.csv(file = dirC[i],header=T,sep=",");
    all<-subset(data)
    set.seed(5)
    nn=sample(c(1:length(all[,1])),length(all[,1]))
    all=cbind(all,nn)
    rmse1<-matrix(0,15,5)
    rmse2<-matrix(0,15,5)
    for(j in 1:5) 
    {
      setIndex=which(all$nn %% 5 == (j-1))
      train<-all[-setIndex,]
      test<-all[setIndex,]
      trainF<-as.data.frame(train$FCH4_F_ANNOPTLM);
      testF<-as.data.frame(test$FCH4_F_ANNOPTLM);
      trainX<-as.data.frame(cbind(train$FCH4_F_ANNOPTLM,train$tmax_y,train$tmin_y,train$dswrf,train$vpd_jz,train$p_jz,train$ws_jz));
      testX<-as.data.frame(cbind(test$FCH4_F_ANNOPTLM,test$tmax_y,test$tmin_y,test$dswrf,test$vpd_jz, test$p_jz,test$ws_jz));
      reg.rf<-randomForest(V1~., data=trainX, ntree=500, mtry =2 )
      
      pred1<-predict(reg.rf, trainX)
      pred2<-predict(reg.rf, testX)
      
      pred1=as.data.frame(pred1)
      pred2=as.data.frame(pred2)
      if(j==1){
        a1=cbind(train$TIMESTAMP,trainF,pred1)
        a2=cbind(test$TIMESTAMP,testF,pred2)
      }else{ 
        temp1=cbind(train$TIMESTAMP,trainF,pred1)
        temp2=cbind(test$TIMESTAMP,testF,pred2)
        a1=rbind(a1,temp1)
        a2=rbind(a2,temp2)
      }
    }
    write.csv(a2,file=paste("D:/seven/results/",d,"d/rf/",substr(fileC[i],1,6),"_rf.csv",sep=""), row.names = F);
  }
  
}