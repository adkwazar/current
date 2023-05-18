from LZcomplexity import LZC
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath 
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from numpy import linalg as LA

class lztree:
    def __init__(self, seqs, names, seq_type = "DNA", tree_method = "average"):
        self.seqs = seqs
        self.names = names
        self.seq_type = seq_type
        self.tree_method = tree_method

    def dist_matrix(self):
        n = len(self.seqs)
        d = dict()
        for i in range(n):
            for j in range(n):
                if i<j:
                    s1 = LZC(self.seqs[i])
                    s2 = LZC(self.seqs[j])
                    sa1 = LZC(self.seqs[i]+self.seqs[j])
                    sa2 = LZC(self.seqs[j]+self.seqs[i])
                    
                    l1 = len(s1.WordSeq())
                    l2 = len(s2.WordSeq())
                    la1 = len(sa1.WordSeq())
                    la2 = len(sa2.WordSeq())
                    
                    m1 = (la1 - min(l1, l2))/max(l1, l2)
                    m2 = (la2 - min(l1, l2))/max(l1, l2)
                    d[(self.names[i], self.names[j])] = (m1+m2)/2
                             
        keys = [sorted(k) for k in d.keys()]
        values = d.values()
        sorted_keys, distances = zip(*sorted(zip(keys, values)))
        dist = linkage(distances, method=self.tree_method)
        labels = sorted(set([key[0] for key in sorted_keys] + [sorted_keys[-1][-1]]))

        return dist, labels
    
        
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
        Z, a = self.dist_matrix()
        Z = to_tree(Z)
        return self.get_newick(Z, Z.dist, self.names)
    
    def plot_tree(self, save = False):
        dist, a = self.dist_matrix()
        fig = plt.figure(figsize=(10, 4))
        dn = dendrogram(dist, labels = a, orientation = "left" , above_threshold_color='black',color_threshold=150)
        plt.show()
        if save is not False:
            fig.savefig('tree.png', dpi=300, bbox_inches="tight")