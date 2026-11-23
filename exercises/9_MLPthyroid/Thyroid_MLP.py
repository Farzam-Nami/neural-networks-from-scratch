from pathlib import Path
import pandas as pd
import numpy as np
import time
inputs_file_path = Path(__file__).resolve().parents[2] / "data" / "thyroid" / "thyroidInputs.xlsx"
targets_file_path = Path(__file__).resolve().parents[2] / "data" / "thyroid" / "thyroidTargets.xlsx"
inputs_file = pd.read_excel(inputs_file_path,header=None)
targets_file = pd.read_excel(targets_file_path,header=None)

inputs_list = inputs_file.T.values.tolist()
targets_list = targets_file.T.values.tolist()

trainingBorder=int(0.7*len(inputs_list))
trainingList=inputs_list[0:trainingBorder]
trainingTargetsList=targets_list[0:trainingBorder]
testBorder=int(0.15*len(inputs_list))
testList=inputs_list[trainingBorder:trainingBorder+testBorder]
testTargetsList=targets_list[trainingBorder:trainingBorder+testBorder]
validationList=inputs_list[trainingBorder+testBorder:]
validationTargetsList=targets_list[trainingBorder+testBorder:]

alpha=0.1
input_count=21
hidden_count=5
output_count=3
v=np.random.uniform(-0.5, 0.5, (input_count,hidden_count))
w=np.random.uniform(-0.5, 0.5, (hidden_count,output_count))
b0=np.random.uniform(-0.5, 0.5, hidden_count)
b1=np.random.uniform(-0.5, 0.5, output_count)


def fNI(yNI):
    """Applies the sigmoid activation function to a neuron input."""
    sigmoid=1/(1+np.exp(-yNI))
    return sigmoid

def fprime_NI(yNI):
    """Computes the sigmoid derivative used during backpropagation."""
    sigmoid_prime=fNI(yNI) * (1 - fNI(yNI))
    return sigmoid_prime
def predict(Xin):
     """Propagates 21 features through 5 hidden and 3 output sigmoid neurons."""
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
def target_find(target_case):
    """Determines the output target used in the thyroid classifier."""
    if(target_case[0]>target_case[1] and target_case[0]>target_case[2]):
         return 1
    elif(target_case[1]>target_case[0] and target_case[1]>target_case[2]):
         return 2
    elif(target_case[2]>target_case[1] and target_case[2]>target_case[0]):
         return 3         
def train(set,targets):
    """Updates the 21-5-3 classifier using targets and validation stopping rules."""
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
                error_o[k]=((targets[i])[k]-y[k])*fprime_NI(Yni[k])
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
        # Use the held-out validation subset for the epoch stopping rule.
        old_loss=loss
        wrongPrediction=0
        for i in range(len(validationList)):
            if(predict(validationList[i])!=target_find(validationTargetsList[i])):
                wrongPrediction+=1
        loss=wrongPrediction              
        new_loss=loss
        if(new_loss<old_loss):
            noChange=0
        elif(new_loss==old_loss):
            noChange+=1

        print("epoch ",epochs)

        epochs+=1
        if(noChange==50):
            stop=True
        if(abs(np.linalg.norm(w-old_w))<0.10 and abs(np.linalg.norm(v-old_v))<0.15 ):
            stop=True
        if(epochs==1000):
            stop=True
      

        
    print("Network trained after ",epochs," epochs")

start = time.time()
train(trainingList,trainingTargetsList)
end = time.time()
print("training took ",(end - start)/60," mins")

with open(Path(__file__).resolve().parent / "results.txt", 'w') as file:
        for i in range(len(w)):
            for j in range(len(w[i])):
                file.write(str(w[i][j]))
                file.write(',')
        file.write('\n')
        for i in range(len(v)):
            for j in range(len(v[i])):
                file.write(str(v[i][j]))
                file.write(',')
        file.write('\n')
        for numbers in b0:
            file.write(str(numbers))
            file.write(',')  
        file.write('\n')
        for numbers in b1:
            file.write(str(numbers))
            file.write(',')      

