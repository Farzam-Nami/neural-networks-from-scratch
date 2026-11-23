from pathlib import Path
import numpy as np
import tkinter as tk
alpha=0.2
input_count=25
hidden_count=8
output_count=1
v=np.random.uniform(-1, 1, (input_count,hidden_count))
w=np.random.uniform(-1, 1, (hidden_count,output_count))
b0=np.random.uniform(-1, 1, hidden_count)
b1=np.random.uniform(-1, 1, output_count)
trainingSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
validationSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "validationset.txt"
# Each sample has 25 bipolar pixels and one class label.
trainingSet=[]
with open(trainingSetPath) as file:
        for line in file:
            trainingSet.append([int(x) for x in line.split()])                 

validationSet=[]
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
    """Runs online backpropagation and checks validation errors per epoch."""
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
        # Count validation errors on all 25 pixels of each held-out pattern.
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
        print("epoch ",epochs)
        
        epochs+=1
        if(noChange==100):
            stop=True
        if(abs(np.linalg.norm(w-old_w))<0.005 and abs(np.linalg.norm(v-old_v))<0.005 and abs(np.linalg.norm(b0-old_b0))<0.005 and abs(np.linalg.norm(b1-old_b1))<0.005):
            stop=True
    print("Network trained after ",epochs," epochs")
    
train(trainingSet)                

#GUI functions
def buttonClick(row, col):
    """Updates the selected cell in the 5-by-5 pattern grid."""
    global changedButtons
    if 0 <= row < 5 and 0 <= col < 5:
        current_color = buttons[row][col].cget("bg")
        if current_color == "blue":
            buttons[row][col].config(bg="white")
            if (row, col) in changedButtons:
                changedButtons.remove((row, col))
        else:
            buttons[row][col].config(bg="blue")
            changedButtons.append((row, col))

def predictButtons():
    """Collects the current grid pattern and evaluates it."""
    global changedButtons
    storedChanges = list(changedButtons)
    lst=[]
    for i in range(5):
        for j in range(5):
            if ((i,j) in storedChanges):
                lst.append(1)
            else:
                lst.append(-1)
    if(predict(lst)==1):
        label_result.config(text="Result: X")
    if(predict(lst)==-1):
        label_result.config(text="Result: O") 
    if(predict(lst)==0):
        label_result.config(text="unknown")                           

def resetButtons():
    """Clears the current pattern from the grid."""
    global changedButtons
    for i in range(5):
        for j in range(5):
            buttons[i][j].config(bg="white")
    changedButtons = []

def storeChanges_x():
    """Stores the current grid pattern as an X training example."""
    global changedButtons
    storedChanges = list(changedButtons)
    resetButtons()
    label_result.config(text="X entered")
    label_guide.config(text="You have to restart for apply changes!")
    lst=[]
    for i in range(5):
        for j in range(5):
            if ((i,j) in storedChanges):
                lst.append(1)
            else:
                lst.append(-1) 
    lst.append(1) 
    with open(trainingSetPath, 'a') as file:
        for numbers in lst:
            file.write(str(numbers)+" ")
        file.write('\n')        
                
def storeChanges_o():
    """Stores the current grid pattern as an O training example."""
    global changedButtons
    storedChanges = list(changedButtons)
    resetButtons()
    label_result.config(text="O entered")
    label_guide.config(text="You have to restart for apply changes!")
    lst=[]
    for i in range(5):
        for j in range(5):
            if ((i,j) in storedChanges):
                lst.append(1)
            else:
                lst.append(-1) 
    lst.append(-1)
    with open(trainingSetPath, 'a') as file:
        for numbers in lst:
            file.write(str(numbers)+" ")
        file.write('\n') 
                       
#Run and use GUI functions
root = tk.Tk()
root.title("Multilayer Perceptron XO")

#Button creator
def createButton(row, col):
    """Creates a button for one cell of the pattern grid."""
    button = tk.Button(root, command=lambda r=row, c=col: buttonClick(r, c), width=10, height=5, bg="white")
    button.grid(row=row, column=col, padx=3, pady=3)
    return button
buttons = [] 
changedButtons = [] 
for i in range(5):
    rowButtons = []
    for j in range(5):
        button = createButton(i, j)
        rowButtons.append(button)
    buttons.append(rowButtons)

reset_button = tk.Button(root, text="Reset", command=resetButtons,width=5)
reset_button.grid(row=6, column=1, pady=10)

x_button = tk.Button(root, text="Enter new X", command=storeChanges_x,width=9)
x_button.grid(row=7, column=0, pady=10)
o_button = tk.Button(root, text="Enter new O", command=storeChanges_o,width=9)
o_button.grid(row=7, column=4, pady=10)

label_result = tk.Label(root, text="",font=10)
label_result.grid(row=7, column=2, pady=10)
label_guide = tk.Label(root, text="",font=5)
label_guide.grid(row=8, column=0,columnspan=5, pady=10)
 
predict_button= tk.Button(root, text="Predict", command=predictButtons,width=5)
predict_button.grid(row=6, column=3, pady=10)

root.mainloop() 