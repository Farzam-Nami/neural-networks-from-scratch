from pathlib import Path
import numpy as np
import tkinter as tk
alpha=0.001 #0.0005<alpha<0.005
w=np.zeros(25)
b=0
trainingSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
# Each sample has 25 bipolar pixels and one class label.
trainingSet=[]
with open(trainingSetPath) as file:
        for line in file:
            trainingSet.append([int(x) for x in line.split()])             

def fY_IN(y):
    """Applies the bipolar activation thresholds to the neuron input."""
    if(y>=0):
        return 1
    elif(y<0):
        return -1

#Main adaline train function   
def train(set):
    """Applies the ADALINE delta rule using the continuous net-input error."""
    global w,b,alpha,epochs
    epochs=0
    stop=False
    while not stop:
        old_w =np.array(w)
        for i in range(len(set)):
            yNI=b
            for j in range(len(w)):
                yNI+=(w[j]*((set[i])[j]))
            for k in range(len(w)):
                w[k]+=alpha*((set[i])[25]-yNI)*(set[i])[k]
            b+=alpha*((set[i])[25]-yNI)
        new_w=np.array(w)
        epochs+=1
        if(np.all(abs(new_w-old_w)<0.001)):
            stop=True
    print("Network trained after ",epochs," epochs")
    output=np.zeros(26)
    output[0:25]=w
    output[25]=b
    with open(Path(__file__).resolve().parent / "adalineXO_results.txt", 'w') as file:
        for numbers in output:
            file.write(str(numbers)+" ")           

#Run train function 
train(trainingSet) 

#predictor function
def predict(Xin):
     """Thresholds the learned linear ADALINE response to X or O."""
     global w,b
     sum=0
     for i in range(len(w)):
        sum=sum+(w[i]*Xin[i])
     sum=sum+b
     return(fY_IN(sum))
           
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
root.title("Adaline XO")

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