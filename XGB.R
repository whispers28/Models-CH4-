library(xgboost)
setwd("D:/") 
for(k in 1:1){
  a=paste("/seven/new/",k,"d") #
  a=gsub(" ",'',a,fixed=TRUE)
  fileC <- list.files(a)            
  dirC <- paste(".",a,"/",fileC,sep="") 
  n<-length(dirC)
  for(i in 1:59)
  {
    data<-read.csv(file = dirC[i],header=T,sep=",");
    data<-subset(data,select = c("TIMESTAMP","tmax_y","tmin_y","dswrf","vpd_jz","ws_jz","p_jz","FCH4_F_ANNOPTLM"))
    all<-subset(data)
    set.seed(5)
    nn=sample(c(1:length(all[,1])),length(all[,1]))
    all=cbind(all,nn)
    rmseTrain<-matrix(1:5,ncol=1)
    rmseTest<-matrix(1:5,ncol=1)
    for(j in 1:5) 
    { 
      setIndex=which(all$nn %% 5 == (j-1))
      train<-all[-setIndex,]
      test<-all[setIndex,]
      trainF<-as.matrix(train$FCH4_F_ANNOPTLM);
      testF<-as.matrix(test$FCH4_F_ANNOPTLM);
      trainX<-as.matrix(cbind(train$tmax_y,train$tmin_y,train$dswrf,train$vpd_jz,train$ws_jz,train$p_jz));
      testX<-as.matrix(cbind(test$tmax_y,test$tmin_y,test$dswrf,test$vpd_jz,test$ws_jz,test$p_jz));      
      set.seed(5)
      bst <- xgboost(trainX, trainF, nrounds =1200, eta =0.05, subsample=0.6 ,verbose = 0, eval_metric='rmse',max_depth=6, lambda=1,min_child_weight=6,nthread = 2)
      pred1<-predict(bst,trainX)
      pred2<-predict(bst,testX)
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
    importance_matrix<-xgb.importance(model=bst)
    write.csv(a2,paste("D:/seven/results/",k,"d/xgb1/",substr(fileC[i],1,6),"_E_test_xgb.csv",sep=""),row.names = FALSE)
    write.csv(importance_matrix,paste("D:/seven/results/importance-1d/",substr(dirC[i],16,21),".csv",sep=""),row.names = FALSE)
  }
}
