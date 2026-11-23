from pathlib import Path
import numpy as np

alpha=0.2
input_count=25
hidden_count=8
output_count=1
v=np.random.uniform(-1, 1, (input_count,hidden_count))
w=np.random.uniform(-1, 1, (hidden_count,output_count))
b0=np.random.uniform(-1, 1, hidden_count)
b1=np.random.uniform(-1, 1, output_count)


validationSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "validationset.txt"
trainSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
testSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "testset.txt"
trainSet=[]
testSet=[]
validationSet=[]
with open(trainSetPath) as file:
        for line in file:
            trainSet.append([int(x) for x in line.split()])
with open(testSetPath) as file:
        for line in file:
            testSet.append([int(x) for x in line.split()])
with open(validationSetPath) as file:
        for line in file:
            validationSet.append([int(x) for x in line.split()])

#Activation Function
def fNI(yNI):
    """Applies the bipolar sigmoid activation function to a neuron input."""
    bipolar_sigmoid=(2/(1+np.exp(-yNI)))-1
    return bipolar_sigmoid
# Derivative used by the backpropagation weight updates. 
def fprime_NI(yNI):
    """Computes the bipolar sigmoid derivative used during backpropagation."""
    bipolar_sigmoid_prime=0.5 * (1 + fNI(yNI)) * (1 - fNI(yNI))
    return bipolar_sigmoid_prime
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
     
     if(y[0]>0):
        return 1
     elif(y[0]<0):
        return -1
     else:
        return 0
#Main mlperceptron train function  
def train(set):
    """Fits the test script's network by online backpropagation."""
    global w,v,b0,b1,alpha,input_count,hidden_count,output_count
    epochs=0
    stop=False
    loss=0
    noChange=0
    while not stop:
        old_w = np.copy(w)
        old_v = np.copy(v)
        old_b0= np.copy(b0)
        old_b1= np.copy(b1)
        for i in range(len(set)):
            Zni=np.zeros(hidden_count)
            z=np.zeros(hidden_count)
            for j in range(hidden_count):
                Zni[j]=b0[j]
                for o in range(input_count):
                    Zni[j]+=(set[i])[o]*((v[o])[j])
                z[j]=fNI(Zni[j])
                
            Yni=np.zeros(output_count)
            y=np.zeros(output_count)
            error_o=np.zeros(output_count)    
            deltaW = np.zeros((hidden_count,output_count))
            deltaB1 = np.zeros(output_count) 
            for k in range(output_count):
                Yni[k]=b1[k]
                for o in range(hidden_count):
                    Yni[k]+=z[o]*((w[o])[k])
                y[k]=fNI(Yni[k])        
                error_o[k]=(((set[i])[25])-y[k])*fprime_NI(Yni[k])
                for o in range(hidden_count):
                    (deltaW[o])[k]=alpha*error_o[k]*z[o]
                deltaB1[k]=alpha*error_o[k]

            deltaV = np.zeros((input_count,hidden_count))
            deltaB0 = np.zeros(hidden_count) 
            D=np.zeros(hidden_count)
            error_h=np.zeros(hidden_count) 
            for j in range(hidden_count):
                for o in range(output_count):
                    D[j]+=error_o[o]*((w[j])[o])
                error_h[j]=D[j]*fprime_NI(Zni[j])
                for o in range(input_count):
                    (deltaV[o])[j]=alpha*error_h[j]*(set[i])[o]
                deltaB0[j]=alpha*error_h[j]

            #updating
            for j in range(hidden_count):
                b0[j]+=deltaB0[j]
                for o in range(input_count):
                    (v[o])[j]+=(deltaV[o])[j]

            for k in range(output_count):
                b1[k]+=deltaB1[k]
                for j in range(hidden_count):
                    (w[j])[k]+=(deltaW[j])[k]        
        # Check predictions against the 26th field (class label).
        old_loss=loss
        wrongPrediction=0
        for i in range(len(validationSet)):
            if(predict((validationSet[i])[0:25])!=(validationSet[i])[25]):
                wrongPrediction+=1
        loss=wrongPrediction               
        new_loss=loss
        if(new_loss<old_loss):
            noChange=0
        elif(new_loss==old_loss):
            noChange+=1
        epochs+=1
        if(noChange==100):
            stop=True
        if(abs(np.linalg.norm(w-old_w))<0.005 and abs(np.linalg.norm(v-old_v))<0.005 and abs(np.linalg.norm(b0-old_b0))<0.005 and abs(np.linalg.norm(b1-old_b1))<0.005):
            stop=True
    print("Network trained after ",epochs," epochs")
    
train(trainSet)
 



trueXcount=0
trueOcount=0
falseXcount=0
falseOcount=0

trainingSetTrueCount=0

for i in range(len(testSet)):
    if(predict(testSet[i][0:25])==1 and (testSet[i])[25]==1):
         trueXcount+=1
    elif(predict(testSet[i][0:25])==1 and (testSet[i])[25]!=1):
         falseXcount+=1
    elif(predict(testSet[i][0:25])==-1 and (testSet[i])[25]==-1):
         trueOcount+=1
    elif(predict(testSet[i][0:25])==-1 and (testSet[i])[25]!=-1):
         falseOcount+=1
         
print("Test results on multi layer perceptron network:\n")
print("True X percentage: ",100*trueXcount/len(testSet))
print("True O percentage: ",100*trueOcount/len(testSet))
print("False X percentage: ",100*falseXcount/len(testSet))
print("False O percentage: ",100*falseOcount/len(testSet))
print("Total true percentage on test set: ",100*(trueXcount+trueOcount)/len(testSet))

for i in range(len(trainSet)):
    if(predict(trainSet[i][0:25])==(trainSet[i])[25]):
            trainingSetTrueCount+=1 

print("\nTest on training set percentage: ",100*trainingSetTrueCount/(len(trainSet)))
