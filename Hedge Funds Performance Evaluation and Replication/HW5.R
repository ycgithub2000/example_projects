setwd("/Users/chenlinxu/Documents/CMU/1-AM/HW5")

factor=read.csv('/Users/chenlinxu/Documents/CMU/1-AM/HW5/factordata.csv',header=TRUE)
#factor=as.matrix(read.csv('/Users/chenlinxu/Documents/CMU/1-AM/HW5/factordata.csv',header=TRUE))
HFindex=as.matrix(read.csv('/Users/chenlinxu/Documents/CMU/1-AM/HW5/HFIndex.csv',header=TRUE))
factor=factor[12:286,]
HFindex=HFindex[1:275,]

numPeriods=275
endIndex=c(93,111,215)

#1
# #dummy variables
# D1=matrix(0,numPeriods,1)
# D1[1:endIndex[1],]=1
# #factor$D1=D1
# D2=matrix(0,numPeriods,1)
# D2[(endIndex[1]+1):endIndex[2],]=1
# #factor$D2=D2
# D3=matrix(0,numPeriods,1)
# D3[(endIndex[2]+1):endIndex[3],]=1
# #factor$D3=D3
# D4=matrix(0,numPeriods,1)
# D4[(endIndex[3]+1):numPeriods,]=1
# #factor$D4=D4
# 
# X=factor[,2:6]
# D1Xt=X
# D1Xt[(endIndex[1]+1):numPeriods,]=0
# D2Xt=X
# D2Xt[c(1:endIndex[1],(endIndex[2]+1):numPeriods),]=0
# D3Xt=X
# D3Xt[c(1:endIndex[2],(endIndex[3]+1):numPeriods),]=0
# D4Xt=X
# D4Xt[1:endIndex[3],]=0

#a c 
#choose 5 indexes:9 10 29 30 39
indexSelected=as.numeric(HFindex[,39])
alpha=matrix(0,4,1)
alpha_stdErr=matrix(0,4,1)
beta=matrix(0,5,4)
beta_stdErr=matrix(0,5,4)
R2_adj=matrix(0,1,4)

for (i in 1:4){
  if(i==1){
    mod=lm(indexSelected[1:endIndex[1]]~(as.matrix(factor[1:endIndex[1],2:6])))
  }
  if(i==2){
    mod=lm(indexSelected[(endIndex[1]+1):endIndex[2]]~(as.matrix(factor[(endIndex[1]+1):endIndex[2],2:6])))
  }
  if(i==3){
    mod=lm(indexSelected[(endIndex[2]+1):endIndex[3]]~(as.matrix(factor[(endIndex[2]+1):endIndex[3],2:6])))
  }
  if(i==4){
    mod=lm(indexSelected[(endIndex[3]+1):numPeriods]~(as.matrix(factor[(endIndex[3]+1):numPeriods,2:6])))
  }
  alpha[i]=coef(summary(mod))[, 1][1]
  alpha_stdErr[i]=coef(summary(mod))[, 2][1]
  beta[1:5,i]=coef(summary(mod))[, 1][2:6]
  beta_stdErr[1:5,i]=coef(summary(mod))[, 2][2:6]
  R2_adj[i]=summary(mod)$adj.r.squared
}
summary(mod)

#b
library(strucchange)
test=sctest(indexSelected ~as.matrix(factor[,2:6]), type= "Nyblom-Hansen", point=3)
test$p.value

#2
#a b
beta_2=matrix(0,5,4)
beta_stdErr_2=matrix(0,5,4)
fittedVal=matrix(0,numPeriods,1)
for (i in 1:4){
  if(i==1){
    mod=lm((indexSelected[1:endIndex[1]]-factor[1:endIndex[1],6])~0+(as.matrix(factor[1:endIndex[1],2:5]-factor[1:endIndex[1],6])))
    fittedVal[1:endIndex[1]]=mod$fitted.values+factor[1:endIndex[1],6]
  }
  if(i==2){
    mod=lm((indexSelected[(endIndex[1]+1):endIndex[2]]-factor[(endIndex[1]+1):endIndex[2],6])~0+(as.matrix(factor[(endIndex[1]+1):endIndex[2],2:5]-factor[(endIndex[1]+1):endIndex[2],6])))
    fittedVal[(endIndex[1]+1):endIndex[2]]=mod$fitted.values+factor[(endIndex[1]+1):endIndex[2],6]
  }
  if(i==3){
    mod=lm((indexSelected[(endIndex[2]+1):endIndex[3]]-factor[(endIndex[2]+1):endIndex[3],6])~0+(as.matrix(factor[(endIndex[2]+1):endIndex[3],2:5]-factor[(endIndex[2]+1):endIndex[3],6])))
    fittedVal[(endIndex[2]+1):endIndex[3]]=mod$fitted.values+factor[(endIndex[2]+1):endIndex[3],6]
  }
  if(i==4){
    mod=lm((indexSelected[(endIndex[3]+1):numPeriods]-factor[(endIndex[3]+1):numPeriods,6])~0+(as.matrix(factor[(endIndex[3]+1):numPeriods,2:5]-factor[(endIndex[3]+1):numPeriods,6])))
    fittedVal[(endIndex[3]+1):numPeriods]=mod$fitted.values+factor[(endIndex[3]+1):numPeriods,6]
  }
  beta_2[1:4,i]=coef(summary(mod))[, 1][2:5]
  beta_2[5,i]=1-sum(coef(summary(mod))[, 1][2:5])
}

#c.i
# gamma=sqrt((indexSelected-mean(indexSelected))^2/(fittedVal-mean(fittedVal))^2)
# R_estimate=matrix(0,numPeriods,1)
# for (i in 1:numPeriods){
#   R_estimate[i]=gamma[i]*fittedVal[i]
# }

gamma=sqrt(sum((indexSelected-mean(indexSelected))^2)/sum((fittedVal-mean(fittedVal))^2))
R_estimate=gamma*fittedVal

delta=1-gamma

TBill=read.csv('/Users/chenlinxu/Documents/CMU/1-AM/HW5/TBill.csv',header=FALSE)
R_hat=R_estimate+delta*TBill[,3]

mean(indexSelected)
sd(indexSelected)
mean(indexSelected)/sd(indexSelected)

mean(R_hat)
sd(R_hat)
mean(R_hat)/sd(R_hat)

#Annualize
mean(indexSelected)*12
sd(indexSelected)*sqrt(12)
mean(indexSelected)/sd(indexSelected)*sqrt(12)

mean(R_hat)*12
sd(R_hat)*sqrt(12)
mean(R_hat)/sd(R_hat)*sqrt(12)



#c.ii
cumret_true=matrix(0,numPeriods,1)
cumret_clone=matrix(0,numPeriods,1)
cumret_true[1]=indexSelected[1]
cumret_clone[1]=R_hat[1]

for (i in 2:numPeriods) {
  cumret_true[i]=(1+cumret_true[i-1])*(1+indexSelected[i])-1
  cumret_clone[i]=(1+cumret_clone[i-1])*(1+R_hat[i])-1
}

#plot(cumret_true,ylim = c(-1,20),xlab = "Time", ylab = "Cumulative Return", xaxt = "n",col=1,type='l')
plot(cumret_clone,xlab = "Time", ylab = "Cumulative Return", xaxt = "n",col=1,type='l')
axis(1, at=seq(1,numPeriods,15), labels=factor[seq(1,numPeriods,15),1])
lines(cumret_true,col = 2) 
legend("topleft", legend=c("Linear Clone","Actual Index"), col=1:2,lty=1,cex=1)





