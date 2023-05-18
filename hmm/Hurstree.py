from CGRepresentation import CGR
from hurst import compute_Hc, random_walk
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath 
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from numpy import linalg as LA

class htree:
    def __init__(self, seqs, names, seq_type = "DNA", tree_method = "average", outer_representation = False, rna_2structure = False):
        self.seqs = seqs
        self.names = names
        self.seq_type = seq_type
        self.tree_method = tree_method
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure

    def hurst(self):
        hs = []
        for seq in self.seqs:
            r = CGR(seq, self.seq_type, self.outer_representation, self.rna_2structure).representation()
            r1 = r[:,0]
            r2 = r[:,1]
            r3 = (r1+r2)/2
            r4 = np.sign(r1*r2)*np.sqrt(np.sign(r1*r2)*(r1*r2))
            
            H1, c, data = compute_Hc(r1, kind='change', simplified=True)
            H2, c, data = compute_Hc(r2, kind='change', simplified=True)
            H3, c, data = compute_Hc(r3, kind='change', simplified=True)
            H4, c, data = compute_Hc(r4, kind='change', simplified=True)
            H = [H1, H2, H3, H4]
            hs.append(H)
        return hs
            
            
    def dist_matrix(self):
        e_s = self.hurst()
        X = np.array(e_s) 
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
  