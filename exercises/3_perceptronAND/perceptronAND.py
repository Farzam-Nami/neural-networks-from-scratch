import matplotlib.pyplot as plt 
import numpy as np
trainingSet=[[1,1,1],[1,0,-1],[0,1,-1],[0,0,-1]]
w1=0
w2=0
b=0
y_NI=0
theta=0.2
alpha=1
weight_list=[]
#training function
def train(vector):
    """Corrects the perceptron weights for one binary-input AND training record."""
    global w1,w2,b,alpha
    y_NI=b+w1*vector[0]+w2*vector[1]
    if(fY_IN(y_NI)!=vector[2]):
        w1=w1+(alpha*vector[0]*vector[2])
        w2=w2+(alpha*vector[1]*vector[2])
        b=b+(alpha*1*vector[2])

def fY_IN(y):
    """Applies the bipolar activation thresholds to the neuron input."""
    global theta
    if(y>theta):
        return 1
    elif(y<-theta):
        return -1
    else:
        return 0

fault=False;    
#Call train function 
while(not fault):
    old_w1=w1
    old_w2=w2
    old_b=b
    for i in range(4):
        train(trainingSet[i])
    weight_list.append([w1,w2,b])    
    if(w1==old_w1 and w2==old_w2 and b==old_b):
        fault=True;    

#showing lines
def showBoundary():
        """Plots the current decision boundary and training patterns."""
        x_P = [1,0,1,0]
        y_P = [1,1,0,0]
        x_L = np.arange(-0.5,2,0.5)
        y_L1 = (-w1/ w2) * x_L + (-b /w2) +(theta/w2)
        y_L2 = (-w1/ w2) * x_L + (-b /w2) +(-theta/w2)
        plt.plot(x_P[1:4], y_P[1:4],'r*')
        plt.plot(x_P[0:1], y_P[0:1],'g*')
        plt.plot(x_L, y_L1,'g')
        plt.plot(x_L, y_L2,'r')
        plt.show()
    
#call show function    
showBoundary()


