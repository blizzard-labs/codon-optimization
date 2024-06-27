import json
from Bio import SeqIO
import random
from collections import OrderedDict

from utils import preproc

#Initializing lists of acceptable DNA bases
dna_seqs = []
dna_seqs_new = []
aa_seqs_new = []
bases = ['A', 'T', 'G', 'C']

for sequence in SeqIO.parse("raw.fna", "fasta"):
    dna_seqs.append(sequence.seq)

for dnaSeq in dna_seqs:
    test_score = 1
    # Tests if divisible by 3 (a set in a codon)
    if len(dnaSeq) %3 != 0:
        test_score = 0
    # Checks if only ATGC
    for base in dnaSeq:
        if base not in bases:
            test_score = 0
    aaSeq = dnaSeq.translate() # Converts to AA
    #AA sequence begins with Met
    if aaSeq[0] != "M":
        test_score = 0
    #AA sequence ends with Stop
    if aaSeq[-1] != "*":
        test_score = 0
    #AA sequence only has one Stop
    if aaSeq.count("*") != 1:
        test_score = 0
    
    if test_score == 1:
        dna_seqs_new.append(str(dnaSeq))
        aa_seqs_new.append(str(aaSeq))

seqDict = {dna_seqs_new[i]: aa_seqs_new[i] for i in range (len(dna_seqs_new))}
items = list(seqDict.items())
random.shuffle(items)
dictShuffled = OrderedDict(items)

f = open("data.json", "w")
json.dump(dictShuffled, f)
f.close()

dnaTrain = list(dictShuffled.keys())[:30000]
aaTrain = list(dictShuffled.values())[:30000]

# Preprocessing DNA and AA Sequences - One Hot Encoding + Padding
x, y = preproc(aaTrain, dnaTrain)

print(x)

f = open("aaTrain.txt", "w")
f.write(x.tolist())
f.close()

f = open("dnaTrain.txt", "w")
f.write(y.tolist())
f.close()
