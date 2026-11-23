from pathlib import Path
import numpy as np
import tkinter as tk

w=np.zeros(25)
b=0
trainingSetPath=Path(__file__).resolve().parents[2] / "data" / "xo" / "trainingset.txt"
# Each sample has 25 bipolar pixels and one class label.
trainingSet=[]
with open(trainingSetPath) as file:
        for line in file:
            trainingSet.append([int(x) for x in line.split()])

#Main hebbian train function                           
def train(set):
    """Accumulates pixel-target products and target values across all X/O samples."""
    global w,b
    for i in range(len(set)):
        x=(set[i])[0:25]
        y=(set[i])[25]
        for j in range(len(w)):
            w[j]=w[j]+(x[j]*y)
        b=b+y
    print("Network trained!")    
    return(w,b)

#Run train function 
w,b=train(trainingSet)

#Writes weights and bias for further use in testing
output=np.zeros(26)
output[0:25]=w
output[25]=b
with open(Path(__file__).resolve().parent / "hebbianXO_results.txt", 'w') as file:
    for numbers in output:
        file.write(str(numbers)+" ") 

#predictor function
def predict(Xin):
     """Classifies an X/O vector using the sign of the Hebbian net input."""
     global w,b
     sum=0
     for i in range(len(w)):
        sum=sum+(w[i]*Xin[i])
     sum=sum+b
     if(sum>0):
          return(1)
     elif(sum<0):
          return(-1)
           
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
root.title("Hebbian XO")

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