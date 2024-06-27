# * 
# * Imports ====================================================================================
# * 

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import aminos, N_ACIDS, N_CODONS, DNA_DEMO, AA_DEMO, nucleics
from utils import preproc

# * 
# * RNN Class Definition ====================================================================================
# * 

class CustomLSTM(nn.Module):
    def __init__(self, inSize=21, n_hidden=128, outSize=64):
        super (CustomLSTM, self).__init__()
        
        self.n_hidden = n_hidden

        self.lstm1 = nn.LSTMCell(inSize, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm4 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.lstm5 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm6 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.dropout3 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, outSize)
        
    def forward (self, x, future=0):
        outputs = []
        n_samples = x.size(0)
        
        h_t1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        h_t3 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t3 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        for idx in range(x.shape[0]):
            currAA = x[i][0]
            
            h_t1, c_t1 = self.lstm1(currAA, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2)) 
            next1 = self.dropout1(h_t2)
            
            h_t3, c_t3 = self.lstm3(next1, (h_t3, c_t3)) 
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4)) 
            next2 = self.dropout1(h_t4)
            
            h_t5, c_t5 = self.lstm5(next2, (h_t5, c_t5)) 
            h_t6, c_t6 = self.lstm6(h_t5, (h_t6, c_t6)) 
            next3 = self.dropout1(h_t6)
            
            output = self.linear1(self.linear2(next3))
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs
            
# * 
# * Model Construction ====================================================================================
# * 

with open('data.json') as f:
    seqDict = json.load(f)

dnaTrain = list(seqDict.keys())[:30000]
aaTrain = list(seqDict.values())[:30000]

# Preprocessing DNA and AA Sequences - One Hot Encoding + Padding
x, y = preproc(aaTrain, dnaTrain)
print(x)

#Finds codon from output
#TODO add translation verification per codon (make sure output DNA is valid candidate for AA)
def codonSelect(output):
    codon_idx = torch.argmax(output).item()
    return nucleics[codon_idx]

model = CustomLSTM()
criterion = nn.MSELoss()
learning_rate = 0.005
optimizer = torch.optim.adam(model.parameters(), lr=learning_rate)

# * 
# * Training Function ====================================================================================
# * 

def train(xTensor, yTensor):
    optimizer.zero_grad()
    
    out = nn.Softmax(model(xTensor))
    
    loss = criterion(out, yTensor)
    print("Loss: ", loss.item())
    
    loss.backward()
    return loss

# * 
# * Training ====================================================================================
# * 

currLoss = 0
losses = []
plot_steps, print_steps = 1000, 5000
epochs = 100000

for i in range(epochs):
    print("Step: " + i)
    optimizer.step(train()) #Add input and output