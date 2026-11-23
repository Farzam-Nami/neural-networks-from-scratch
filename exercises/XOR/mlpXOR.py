from matplotlib import pyplot as plt
import numpy as np

trainingSet=[[1,1,-1],[1,-1,1],[-1,1,1],[-1,-1,-1]]
alpha=0.1
input_count=2
hidden_count=2
output_count=1
v=[[[0.5],[0.5]],[[-0.5],[-0.5]]]#np.random.uniform(0, 1, (input_count,hidden_count))
w=np.random.uniform(0, 1, (hidden_count,output_count))
b0=[0.5,-0.5]
b1=[0.5]
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
    """Trains a 2-2-1 bipolar sigmoid network for the four XOR examples."""
    global w,v,b0,b1,alpha,input_count,hidden_count,output_count
    epochs=0
    stop=False
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
                error_o[k]=(((set[i])[2])-y[k])*fprime_NI(Yni[k])
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

        print("epoch ",epochs)
        
        epochs+=1
        
        if(abs(np.linalg.norm(w-old_w))<0.01 and abs(np.linalg.norm(v-old_v))<0.01 and abs(np.linalg.norm(b0-old_b0))<0.01 and abs(np.linalg.norm(b1-old_b1))<0.01):
            stop=True
    print("Network trained after ",epochs," epochs")
    
train(trainingSet)
print("w",w)
print("v",v)
print("b0",b0)
print("b1",b1)


def showBoundary():
        """Plots the current decision boundary and training patterns."""
        x_P = [1,0,1,0]
        y_P = [1,1,0,0]
        x_L = np.arange(-0.5,2,0.5)
        y_L1 = (-v[0][0]/ v[0][1]) * x_L + (-b0[0] /v[0][1]) 
        y_L2 = (-v[1][0]/ v[1][1]) * x_L + (-b0[1] /v[1][1]) 
        plt.plot(x_P[0], y_P[0],'r*')
        plt.plot(x_P[3], y_P[3],'r*')
        plt.plot(x_P[1:3], y_P[1:3],'g*')
        plt.plot(x_L, y_L1,'r')
        plt.plot(x_L, y_L2,'r')
        plt.show()
    
#call show function    
showBoundary()