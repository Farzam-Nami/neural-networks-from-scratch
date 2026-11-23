from pathlib import Path
with open(Path(__file__).resolve().parent / "hebbianXO_results.txt") as file:
      for line in file:
            weights_bias=[float(x) for x in line.split()]
w=weights_bias[0:25]
b=weights_bias[25]
trainSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
testSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "testset.txt"
trainSet=[]
testSet=[]
with open(trainSetPath) as file:
        for line in file:
            trainSet.append([int(x) for x in line.split()])
with open(testSetPath) as file:
        for line in file:
            testSet.append([int(x) for x in line.split()])
def predict(Xin):
     """Computes the model prediction for an input pattern using the current weights."""
     global w,b
     sum=0
     for i in range(len(w)):
        sum=sum+(w[i]*Xin[i])
     sum=sum+b
     if(sum>0):
          return(1)
     elif(sum<0):
          return(-1)
     
trueXcount=0
trueOcount=0
falseXcount=0
falseOcount=0

trainingSetTrueCount=0

for i in range(len(testSet)):
    if(predict(testSet[i])==1 and (testSet[i])[25]==1):
         trueXcount+=1
    elif(predict(testSet[i])==1 and (testSet[i])[25]!=1):
         falseXcount+=1
    elif(predict(testSet[i])==-1 and (testSet[i])[25]==-1):
         trueOcount+=1
    elif(predict(testSet[i])==-1 and (testSet[i])[25]!=-1):
         falseOcount+=1             
print("Test results on hebbian network:\n")
print("True X percentage: ",100*trueXcount/len(testSet))
print("True O percentage: ",100*trueOcount/len(testSet))
print("False X percentage: ",100*falseXcount/len(testSet))
print("False O percentage: ",100*falseOcount/len(testSet))
print("Total true percentage on test set: ",100*(trueXcount+trueOcount)/len(testSet))

for i in range(len(trainSet)):
    if(predict(trainSet[i])==(trainSet[i])[25]):
            trainingSetTrueCount+=1 

print("\nTest on training set percentage: ",100*trainingSetTrueCount/(len(trainSet)))                  
