# * 
# * Imports =================================================================================================
# * 

from io import open
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Seq import translate
from Bio.SeqRecord import SeqRecord

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

#Utilize CUDA cores
def initCuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# * 
# * Managing Data Files =====================================================================================
# * 

N_CODONS = 64
N_ACIDS = 21
AA_DEMO = 'MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTFIIRCLQWTTVIERTFHVETPEEREEWTTAIQTVADGLKKQEEEEMDFRSGSPSDNSGAEEMEVSLAKPKHRVTMNEFEYLKLLGKGTFGKVILVKEKATGRYYAMKILKKEVIVAKDEVAHTLTENRVLQNSRHPFLTALKYSFQTHDRLCFVMEYANGGELFFHLSRERVFSEDRARFYGAEIVSALDYLHSEKNVVYRDLKLENLMLDKDGHIKITDFGLCKEGIKDGATMKTFCGTPEYLAPEVLEDNDYGRAVDWWGLGVVMYEMMCGRLPFYNQDHEKLFELILMEEIRFPRTLGPEAKSLLSGLLKKDPKQRLGGGSEDAKEIMQHRFFAGIVWQHVYEKKLSPPFKPQVTSETDTRYFDEEFTAQMITITPPDQDDSMECVDSERRPHFPQFSYSASGTA*'
DNA_DEMO = 'ATGAGCGACGTGGCTATTGTGAAGGAGGGTTGGCTGCACAAACGAGGGGAGTACATCAAGACCTGGCGGCCACGCTACTTCCTCCTCAAGAATGATGGCACCTTCATTGGCTACAAGGAGCGGCCGCAGGATGTGGACCAACGTGAGGCTCCCCTCAACAACTTCTCTGTGGCGCAGTGCCAGCTGATGAAGACGGAGCGGCCCCGGCCCAACACCTTCATCATCCGCTGCCTGCAGTGGACCACTGTCATCGAACGCACCTTCCATGTGGAGACTCCTGAGGAGCGGGAGGAGTGGACAACCGCCATCCAGACTGTGGCTGACGGCCTCAAGAAGCAGGAGGAGGAGGAGATGGACTTCCGGTCGGGCTCACCCAGTGACAACTCAGGGGCTGAAGAGATGGAGGTGTCCCTGGCCAAGCCCAAGCACCGCGTGACCATGAACGAGTTTGAGTACCTGAAGCTGCTGGGCAAGGGCACTTTCGGCAAGGTGATCCTGGTGAAGGAGAAGGCCACAGGCCGCTACTACGCCATGAAGATCCTCAAGAAGGAAGTCATCGTGGCCAAGGACGAGGTGGCCCACACACTCACCGAGAACCGCGTCCTGCAGAACTCCAGGCACCCCTTCCTCACAGCCCTGAAGTACTCTTTCCAGACCCACGACCGCCTCTGCTTTGTCATGGAGTACGCCAACGGGGGCGAGCTGTTCTTCCACCTGTCCCGGGAGCGTGTGTTCTCCGAGGACCGGGCCCGCTTCTATGGCGCTGAGATTGTGTCAGCCCTGGACTACCTGCACTCGGAGAAGAACGTGGTGTACCGGGACCTCAAGCTGGAGAACCTCATGCTGGACAAGGACGGGCACATTAAGATCACAGACTTCGGGCTGTGCAAGGAGGGGATCAAGGACGGTGCCACCATGAAGACCTTTTGCGGCACACCTGAGTACCTGGCCCCCGAGGTGCTGGAGGACAATGACTACGGCCGTGCAGTGGACTGGTGGGGGCTGGGCGTGGTCATGTACGAGATGATGTGCGGTCGCCTGCCCTTCTACAACCAGGACCATGAGAAGCTTTTTGAGCTCATCCTCATGGAGGAGATCCGCTTCCCGCGCACGCTTGGTCCCGAGGCCAAGTCCTTGCTTTCAGGGCTGCTCAAGAAGGACCCCAAGCAGAGGCTTGGCGGGGGCTCCGAGGACGCCAAGGAGATCATGCAGCATCGCTTCTTTGCCGGTATCGTGTGGCAGCACGTGTACGAGAAGAAGCTCAGCCCACCCTTCAAGCCCCAGGTCACGTCGGAGACTGACACCAGGTATTTTGATGAGGAGTTCACGGCCCAGATGATCACCATCACACCACCTGACCAAGATGACAGCATGGAGTGTGTGGACAGCGAGCGCAGGCCCCACTTCCCCCAGTTCTCCTACTCGGCCAGCGGCACGGCCTGA'

aminos = {
    'A': [0,  ['GCA', 'GCC', 'GCG', 'GCT']],
    'R': [1,  ['CGG', 'CGT', 'CGA', 'CGC', 'AGA', 'AGG']],
    'N': [2,  ['AAC', 'AAT']],
    'D': [3,  ['GAT', 'GAC']],
    'C': [4,  ['TGC', 'TGT']],
    'Q': [5,  ['CAA', 'CAG']],
    'E': [6,  ['GAA', 'GAG']],
    'G': [7,  ['GGG', 'GGA', 'GGC', 'GGT']],
    'H': [8,  ['CAC', 'CAT']],
    'I': [9,  ['ATA', 'ATC', 'ATT']],
    'L': [10, ['CTA', 'CTC', 'CTT', 'CTG', 'TTA', 'TTG']],
    'K': [11, ['AAA', 'AAG']],
    'M': [12, ['ATG']],
    'F': [13, ['TTC', 'TTT']],
    'P': [14, ['CCG', 'CCT', 'CCA', 'CCC']],
    'S': [15, ['AGC', 'AGT', 'TCA', 'TCG', 'TCT', 'TCC']],
    'T': [16, ['ACA', 'ACG', 'ACT', 'ACC']],
    'W': [17, ['TGG']],
    'Y': [18, ['TAC', 'TAT']],
    'V': [19, ['GTA', 'GTC', 'GTG', 'GTT']],
    '*': [20, ['TGA', 'TAG', 'TAA']],
    }

nucleics = ['AAA', 'AAC','AAG','AAT','ACA','ACG','ACT','AGC','ATA','ATC','ATG','ATT','CAA','CAC','CAG','CCG','CCT','CTA','CTC','CTG','CTT','GAA','GAT','GCA','GCC','GCG','GCT','GGA','GGC','GTC','GTG','GTT','TAA','TAT','TCA','TCG','TCT','TGG','TGT','TTA','TTC','TTG','TTT','ACC','CAT','CCA','CGG','CGT','GAC','GAG','GGT','AGT','GGG','GTA','TGC','CCC','CGA','CGC','TAC','TAG','TCC','AGA','AGG','TGA']

#Translating DNA to an AA sequence
def dna2aa (dna):
    return (Seq(dna).translate())

# * 
# * Data Preprocessing ======================================================================================
# * 

#Converting AA sequence to a list of integer labels (One hot encoding)
def aa2tensor(aa):
    seq = [aminos[i][0] for i in aa]
    tensor = torch.zeros(len(seq), 1,N_ACIDS)
    for i, aa in enumerate(seq):
        tensor[i][0][aa] = 1
    return tensor

#One-hot encoding DNA sequence to list of integer labels
def dna2tensor(dna):
    tensor = torch.zeros(len(dna), 1, N_CODONS)
    for i, codon in enumerate(dna):
        tensor[i][0][nucleics.index(codon)] = 1
    return(tensor)

#Encrypting DNA and AA sequences into "words" 
# ! Function not in use
def encrypt (string, length):
    return ' '.join(string[i:i+length] for i in range(0, len(string), length))

#Add padding to sequences that are shorter than required length
def pad (sequences):
    return (pad_sequence(sequences, batch_first=True))

# Preprocessing Steps Combined | x = AA; y = DNA
def preproc (x, y):
    xTensor = pad([aa2tensor(seq) for seq in x])
    
    ySplit = [] #Spliting every 3rd nucleotide to codons for encoding
    # Ex. ["ATGTGC", "TGC"] --> [["ATG", "TGC"], ["TGC"]]
    #        S1        S2          S1C1   S1C2     S2C1 
    for seq in y:
        ySplit.append([seq[i:i+3] for i in range(0, len(seq), 3)])
    
    yTensor = pad([dna2tensor(seq) for seq in ySplit])
    
    return xTensor, yTensor

# * 
# * Testing Scratchboard ====================================================================================
# * 

if __name__ == '__main__':
    x, y = preproc(["ARN", "NA"], ["ATGTGC", "TGC"])
    print(x)
    print(x.shape)
    '''
    print(y.shape)
    print(x.shape)
    print(x[0].shape)
    print(x[0].size())
    print(x[0])
    #print(dna2tensor(["ATG", ""]))'''