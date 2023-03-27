DNA_2 = {'G': { 'G': 1, 'C':-3, 'A':-3, 'T':-3, 'N':0 },
'C': { 'G':-3, 'C': 1, 'A':-3, 'T':-3, 'N':0 },
'A': { 'G':-3, 'C':-3, 'A': 1, 'T':-3, 'N':0 },
'T': { 'G':-3, 'C':-3, 'A':-3, 'T': 1, 'N':0 },
'N': { 'G': 0, 'C': 0, 'A': 0, 'T': 0, 'N':0 }}


def SequenceAlign(seqA, seqB, similarityMatrix=DNA_2, insert=5, extend=3):
    
    import numpy as np
    
    numI = len(seqA) + 1
    numJ = len(seqB) + 1
    
    SMatrix = np.zeros((numI, numJ))
    RMatrix = np.zeros((numI, numJ))
    
    for i in range(1, numI):
        RMatrix[i, 0] = 1
        
    for j in range(1, numJ):
        RMatrix[0, j] = 2
    
    for i in range(1, numI):
        for j in range(1, numJ):
            
            penalty1 = insert
            penalty2 = insert
            
            if RMatrix[i-1, j] == 1:
                penalty1 = extend
                
            elif RMatrix[i, j-1] == 2:
                penalty2 = extend
                
            similarity = similarityMatrix[seqA[i-1]][seqB[j-1]]
            
            paths = [SMatrix[i-1, j-1] + similarity,
                     SMatrix[i-1, j] - penalty1,
                     SMatrix[i, j-1] - penalty2]
        
            best = max(paths)         #maximum value of path list
            route = paths.index(best) #index where maximum value
        
            SMatrix[i, j] = best  
            RMatrix[i, j] = route
                    
        alignA = []
        alignB = []
        
        i = numI-1
        j = numJ-1
            
        score = SMatrix[i, j]
        
        while i > 0 or j > 0:
            route = RMatrix[i, j]
            
            if route == 0: 
                alignA.append( seqA[i-1] )
                alignB.append( seqB[j-1] )
                i -= 1
                j -= 1
                
            elif route == 1:
                alignA.append( seqA[i-1] )
                alignB.append( '-' )
                i -= 1
                
            elif route == 2: 
                alignA.append( '-' )
                alignB.append( seqB[j-1] )
                j -= 1
                
    alignA.reverse()
    alignB.reverse()
    
    alignA = ''.join(alignA)
    alignB = ''.join(alignB)
    
    return score, alignA, alignB 


import numpy as np
import matplotlib.pyplot as plt


def dotMatrix(sequence1, sequence2):
    
    S1 = len(sequence1)
    S2 = len(sequence2)

    A = np.zeros((S1, S2))

    for i in range(S1):
        for j in range(S2):
            if sequence1[i] == sequence2[j]:  #if sequences have the same residue at i and j poistions respectively
                A[i,j] = 1

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(A)
    xaxis = np.arange(S2)
    yaxis = np.arange(S1)
    ax.set_xticks(xaxis)
    ax.set_yticks(yaxis)
    ax.set_xticklabels(sequence2)
    ax.set_yticklabels(sequence1)

    plt.show()