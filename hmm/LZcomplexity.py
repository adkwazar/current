import numpy as np
import matplotlib.pyplot as plt


class LZC:
    def __init__(self, seq):
        self.seq = seq
               
    def WordSeq(self):
        d=[]
        l=len(self.seq)
        i=0
        k=1
        n=0
        while i<l:
            while self.seq[i:i+k] in d and i+k<l:
                k+=1
            if self.seq[i:i+k] not in d:     
                d.append(self.seq[i:i+k]) 
            i+=k  
            k=1  
        return d