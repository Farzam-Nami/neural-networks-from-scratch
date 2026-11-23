from pathlib import Path
import numpy as np
import tkinter as tk
alpha=0.1
theta=0.5
outputCounts=2
w=[np.zeros(25),np.zeros(25)]
b=np.zeros(outputCounts)
trainingSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
# Each sample has 25 bipolar pixels and one class label.
trainingSet=[]
with open(trainingSetPath) as file:
        for line in file:
            trainingSet.append([int(x) for x in line.split()])    

def fY_IN(y):
    """Applies the bipolar activation thresholds to the neuron input."""
    global theta
    if(y>theta):
        return 1
    elif(y<-theta):
        return -1
    else:
        return 0
    
def train(set):
    """Trains two output neurons on the complementary X/O target pair."""
    global w,b,alpha,epochs,outputCounts
    epochs=0
    stop=False
    while not stop:
        old_w=np.array(w)
        old_b=np.array(b)
        for i in range(len(set)):
            yNI=[b[0],b[1]]
            targets=[(set[i])[25],-(set[i])[25]]

            for k in range(len(yNI)):
                for j in range(len(w[0])):
                    yNI[k]+=((w[k])[j]*((set[i])[j]))
            for k in range(len(targets)):
                if(fY_IN(yNI[k])!=targets[k]):
                    for j in range(len(w[0])):
                        (w[k])[j]+=alpha*((set[i])[j])*(targets[k])
                    b[k]+=alpha*targets[k]
    
        new_w=np.array(w)
        new_b=np.array(b)
        epochs+=1
        if(np.array_equal(new_w,old_w) and np.array_equal(new_b,old_b)):
            stop=True
    print("Network trained after ",epochs," epochs")
    output=[np.zeros(26),np.zeros(26)]
    (output[0])[0:25]=w[0]
    (output[1])[0:25]=w[1]
    (output[0])[25]=b[0]
    (output[1])[25]=b[1]
    with open(Path(__file__).resolve().parent / "MCperceptronXO_results.txt", 'w') as file:
        for o in range(len(output)):
            for numbers in output[o]:
                file.write(str(numbers)+" ")
            file.write("\n")               
train(trainingSet)                

    #predictor function
def predict(Xin):
     """Combines both output neurons; returns 0 for a non-complementary response."""
     global w,b
     sum0=b[0]
     sum1=b[1]
     for i in range(len(w[0])):
        sum0+=((w[0])[i]*Xin[i])
        sum1+=((w[1])[i]*Xin[i])
     if(fY_IN(sum0)==1 and fY_IN(sum1)==-1):
         return 1
     elif(fY_IN(sum0)==-1 and fY_IN(sum1)==1):
         return -1
     else:
         return 0
    
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
root.title("Multicatecory Perceptron XO")

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