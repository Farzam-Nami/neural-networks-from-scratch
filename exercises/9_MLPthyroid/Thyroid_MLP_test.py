from pathlib import Path
import pandas as pd
import numpy as np
resultsPath = Path(__file__).resolve().parent / "results.txt"
inputsFilePath = Path(__file__).resolve().parents[2] / "data" / "thyroid" / "thyroidInputs.xlsx"
targetsFilePath = Path(__file__).resolve().parents[2] / "data" / "thyroid" / "thyroidTargets.xlsx"
input_count=21
hidden_count=5
output_count=3
#Read lines from the file
linesAsLists = []
with open(resultsPath, 'r') as file:
    for line in file:
        line = line.strip()
        numbersList = line.split(',')
        numbersList = [float(num) if num != '' else None for num in numbersList]
        numbersList = [num for num in numbersList if num is not None]
        linesAsLists.append(numbersList)

#lists obtained from the file
w=np.zeros((hidden_count,output_count))
v=np.zeros((input_count,hidden_count))
for i in range(hidden_count):
    w[i]=linesAsLists[0][i*output_count:(i+1)*output_count]        
for i in range(input_count):
    v[i]=linesAsLists[1][i*hidden_count:(i+1)*hidden_count]
b0=linesAsLists[2]
b1=linesAsLists[3]        

#test list
inputsFile = pd.read_excel(inputsFilePath,header=None)
targetsFile = pd.read_excel(targetsFilePath,header=None)
inputsList = inputsFile.T.values.tolist()
targetsList = targetsFile.T.values.tolist()
trainingBorder=int(0.7*len(inputsList))
testBorder=int(0.15*len(inputsList))
testList=inputsList[trainingBorder:trainingBorder+testBorder]
testTargetsList=targetsList[trainingBorder:trainingBorder+testBorder]


def fNI(yNI):
    """Applies the sigmoid activation function to a neuron input."""
    sigmoid=1/(1+np.exp(-yNI))
    return sigmoid
def predict(Xin):
     """Computes the model prediction for an input pattern using the current weights."""
     global w,v,b0,b1

     zIN=np.zeros(len(b0))
     z=np.zeros(len(b0))
     for j in range(len(b0)):
         zIN[j]=b0[j]
         for i in range(len(Xin)):
             zIN[j]+=Xin[i]*((v[i])[j])
         z[j]=fNI(zIN[j])

     yIN=np.zeros(len(b1))
     y=np.zeros(len(b1))
     for k in range(len(b1)):
         yIN[k]=b1[k]
         for j in range(len(b0)):
             yIN[k]+=z[j]*((w[j])[k])
         y[k]=fNI(yIN[k])    
     if(y[0]>y[1] and y[0]>y[2]):
         return 1
     elif(y[1]>y[0] and y[1]>y[2]):
         return 2
     elif(y[2]>y[1] and y[2]>y[0]):
         return 3

def targetFind(targetCase):
    """Identifies the target class from the network output."""
    if(targetCase[0]>targetCase[1] and targetCase[0]>targetCase[2]):
         return 1
    elif(targetCase[1]>targetCase[0] and targetCase[1]>targetCase[2]):
         return 2
    elif(targetCase[2]>targetCase[1] and targetCase[2]>targetCase[0]):
         return 3


pNormal_rNormal_Count=0
pNormal_rHyper_Count=0
pNormal_rSub_Count=0
pHyper_rNormal_Count=0
pHyper_rHyper_Count=0
pHyper_rSub_Count=0
pSub_rNormal_Count=0
pSub_rHyper_Count=0
pSub_rSub_Count=0

for i in range(len(testList)):
    if(predict(testList[i])==1 and targetFind(testTargetsList[i])==1):
        pNormal_rNormal_Count+=1
    elif(predict(testList[i])==1 and targetFind(testTargetsList[i])==2):
        pNormal_rHyper_Count+=1
    elif(predict(testList[i])==1 and targetFind(testTargetsList[i])==3):
        pNormal_rSub_Count+=1           
    elif(predict(testList[i])==2 and targetFind(testTargetsList[i])==1):
        pHyper_rNormal_Count+=1
    elif(predict(testList[i])==2 and targetFind(testTargetsList[i])==2):
        pHyper_rHyper_Count+=1
    elif(predict(testList[i])==2 and targetFind(testTargetsList[i])==3):
        pHyper_rSub_Count+=1           
    elif(predict(testList[i])==3 and targetFind(testTargetsList[i])==1):
        pSub_rNormal_Count+=1
    elif(predict(testList[i])==3 and targetFind(testTargetsList[i])==2):
        pSub_rHyper_Count+=1
    elif(predict(testList[i])==3 and targetFind(testTargetsList[i])==3):
        pSub_rSub_Count+=1
    

print("\n\n\nTest results on 2 layer thyroid network:\n")

print("predicted Normal-Normal percentage: ",100*pNormal_rNormal_Count/len(testList))
print("predicted Normal-Hyperfunction  percentage: ",100*pNormal_rHyper_Count/len(testList))
print("predicted Normal-Subnormal  percentage: ",100*pNormal_rSub_Count/len(testList))

print("\npredicted Hyperfunction-Normal percentage: ",100*pHyper_rNormal_Count/len(testList))
print("predicted Hyperfunction-Hyperfunction  percentage: ",100*pHyper_rHyper_Count/len(testList))
print("predicted Hyperfunction-Subnormal  percentage: ",100*pHyper_rSub_Count/len(testList))

print("\npredicted Subnormal-Normal percentage: ",100*pSub_rNormal_Count/len(testList))
print("predicted Subnormal-Hyperfunction  percentage: ",100*pSub_rHyper_Count/len(testList))
print("predicted Subnormal-Subnormal  percentage: ",100*pSub_rSub_Count/len(testList))

print("\n\nTotal true percentage on test set: ",100*(pNormal_rNormal_Count+pHyper_rHyper_Count+pSub_rSub_Count)/len(testList))
print("\n\n\n")