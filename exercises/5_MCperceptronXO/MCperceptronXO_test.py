from pathlib import Path
with open(Path(__file__).resolve().parent / "MCperceptronXO_results.txt") as file:
      model_rows = [[float(x) for x in line.split()] for line in file if line.strip()]
w = [row[0:25] for row in model_rows[:2]]
b = [row[25] for row in model_rows[:2]]
theta=0.5
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
     scores = [sum(w[k][i] * Xin[i] for i in range(25)) + b[k] for k in range(2)]
     if scores[0] > theta and scores[1] < -theta:
        return 1
     if scores[1] > theta and scores[0] < -theta:
        return -1
     return 0

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
         
print("Test results on multicategory perceptron network:\n")
print("True X percentage: ",100*trueXcount/len(testSet))
print("True O percentage: ",100*trueOcount/len(testSet))
print("False X percentage: ",100*falseXcount/len(testSet))
print("False O percentage: ",100*falseOcount/len(testSet))
print("Total true percentage on test set: ",100*(trueXcount+trueOcount)/len(testSet))

for i in range(len(trainSet)):
    if(predict(trainSet[i])==(trainSet[i])[25]):
            trainingSetTrueCount+=1 

print("\nTest on training set percentage: ",100*trainingSetTrueCount/(len(trainSet)))
                  
