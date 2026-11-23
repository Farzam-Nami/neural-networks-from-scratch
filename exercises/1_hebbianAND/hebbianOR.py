import matplotlib.pyplot as plt 
import numpy as np
trainingSet=[[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,-1]]
w1=0
w2=0
b=0
decisionBoundaries=[]

#training function
def train(vector):
    """Updates the Hebbian weights for one bipolar OR sample and stores its boundary."""
    new_w1=w1+(vector[0]*vector[2])
    new_w2=w2+(vector[1]*vector[2])
    new_b=b+(1*vector[2])
    #finding boundaries
    decisionBoundaries.append([new_w1,new_w2,new_b])
    newSpec=[new_w1,new_w2,new_b]
    return(newSpec)

#Call train function 
for i in range(4):
    newSpec=train(trainingSet[i])
    w1=newSpec[0]
    w2=newSpec[1]
    b=newSpec[2]

#showing lines
def showBoundaries():
    """Plots the decision boundaries after the weight updates."""
    for i in range(4):
        x_P = [1,-1, 1,-1]
        y_P = [1, 1,-1,-1]
        x_L = np.arange(-1.5,2,0.5)
        if decisionBoundaries[i][1] == 0:
            # A vertical decision boundary cannot be written as y = mx + b.
            if decisionBoundaries[i][0] != 0:
                plt.axvline(-decisionBoundaries[i][2] / decisionBoundaries[i][0])
        else:
            y_L = (-decisionBoundaries[i][0] / decisionBoundaries[i][1]) * x_L + (-decisionBoundaries[i][2] / decisionBoundaries[i][1])
            plt.plot(x_L, y_L)
        plt.plot(x_P[1:4], y_P[1:4],'r*')
        plt.plot(x_P[0:1], y_P[0:1],'g*')
        if(i==3):
            plt.title(i+1)
        else:    
            plt.title(str(i+1)+"     close window to see next line ->")
        plt.show()
    
#call show function    
showBoundaries()

