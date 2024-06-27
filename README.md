# codon-optimization
Codon optimization for RNA and DNA based protein manufacturing. 

Built with bi-directional LSTMS
* Pytorch
* CUDA Development

There are 64 codons, but only 21 amino acids. When developing sequences, researchers often use the most occuring codon to code for a specific amino acid. This model optimizes the nucleotide sequence to match the model organism's natural sequence.

Model Organism: Chinese Hampster

Currently still a work in progress (training and decoding is incomplete).

TODO:
* Training and Packaging Model
* Decoding output sequence

Future:
* Migration to MAMBA State Space Models
* Further integration of other model organisms genomes (e. coli?)
