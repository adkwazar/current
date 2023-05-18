from CGRepresentation import CGR
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath 
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from numpy import linalg as LA

class dtree:
    def __init__(self, seqs, names, seq_type = "DNA", tree_method = "average", outer_representation = False, rna_2structure = False):
        self.seqs = seqs
        self.names = names
        self.seq_type = seq_type
        self.tree_method = tree_method
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure
        
    def power_spectra(self):
        spectra = []
        for seq in self.seqs:
            r = CGR(seq, self.seq_type, self.outer_representation, self.rna_2structure).representation()
            F = fft([complex(x[0],x[1]) for x in r][1:])
            spectra.append([abs(elem)**2 for elem in F])
        return spectra
    
    def elongated_spectra(self):
        m = max([len(seq) for seq in self.seqs])
        spectra = self.power_spectra()
        elongated_spectra = []
        for s in spectra:
            n = len(s)
            if n == m :
                elongated_spectra.append(s[1:])
            else:
                s2 = []
                s2.append(s[0])
                for k in range(1,m):
                    Q = k*n/m
                    R = math.floor(Q)
                    if R == 0:
                        R = 1
                        
                    if (Q).is_integer():
                        s2.append(s[int(Q)])
                    else:
                        if R<n-1:
                            s2.append(s[R]+(Q-R)*(s[R+1]-s[R]))
                        else:
                            s2.append(s[R])
                elongated_spectra.append(s2[1:])
                
        return elongated_spectra
    
    def dist_matrix(self):
        e_s = self.elongated_spectra()
        X = np.array([elem[1:] for elem in e_s]) 
        dist = linkage(X, method=self.tree_method)
        return dist
    
        
    def get_newick(self, node, parent_dist, leaf_names, newick =" "):
        if node.is_leaf():
            return "%s%s" % (leaf_names[node.id], newick)
        else:
            if len(newick) > 0 and newick != " ":
                newick = ")%s" % (newick)
            else:
                newick = ");"
            newick = self.get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
            newick = self.get_newick(node.get_right(), node.dist, leaf_names, newick=",%s" % (newick))
            newick = "(%s" % (newick)
            return newick
            
    def newick(self):
        Z = self.dist_matrix()
        Z = to_tree(Z)
        return self.get_newick(Z, Z.dist, self.names)
    
    def plot_tree(self, save = False):
        dist = self.dist_matrix()
        fig = plt.figure(figsize=(10, 4))
        dn = dendrogram(dist, labels = self.names, orientation = "left" , above_threshold_color='black',color_threshold=150)
        plt.show()
        if save is not False:
            fig.savefig('tree.png', dpi=300, bbox_inches="tight")
  