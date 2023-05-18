from CGRepresentation import CGR
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath 
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from numpy import linalg as LA

class stree:
    def __init__(self, seqs, names, seq_type = "DNA", tree_method = "average", outer_representation = False, rna_2structure = False):
        self.seqs = seqs
        self.names = names
        self.seq_type = seq_type
        self.tree_method = tree_method
        self.outer_representation = outer_representation
        self.rna_2structure = rna_2structure
    
    
    def dist_matrix(self):
        n = len(self.seqs)
        for i in range(n):
            r = CGR(self.seqs[i], self.seq_type, self.outer_representation, self.rna_2structure)
            r.plot(axis = False, save = True, im_name = self.names[i], show = False)
        d = dict()
        for i in range(n):
            for j in range(n):
                if i<j:
                    imageA = cv2.imread(f"{self.names[i]}.png")
                    imageB = cv2.imread(f"{self.names[j]}.png")
                    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
                    (score, diff) = structural_similarity(grayA, grayB, full=True)
                    d[(self.names[i], self.names[j])] = 1 - score
                    
                    
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
  