import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#otwieram plik z mikromacierzą i sortuje wg pierwszej kolumny
df = pd.read_csv('yeast_expression.txt', sep='\s+')
df = df.drop(['GENE'], axis=1)
df2 = df.sort_values(by = 'T0') #sortuje wg pierwszej kolumny


#wizualizacja
plt.figure(figsize=(6,10))
plt.matshow(df2, fignum=1, aspect='auto')
plt.show()


#Redukcja 7 kolumn do dwoch cech - PC1/PC2
sc = StandardScaler()
Xs = sc.fit_transform(df)
pca = PCA(n_components=2) #2 cechy
pca.fit(Xs)
X_pca = pca.transform(Xs)


plt.figure(figsize=(15,9))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()



#Jaki procent informacji o zmiennosci danych pozostal po pozostawieniu 2 cech
pca = PCA(n_components=None)
X_pca = pca.fit_transform(Xs)
print("Po zredukowaniu liczby cech do 2, pozostało ", round(sum(pca.explained_variance_ratio_[:2])*100), "% informacji o zmienności danych")



#Pogrupujmy te dane (geny) na 2 grupy
km = KMeans(n_clusters=2,random_state=123) #podzial na 2 grupy
y = km.fit_predict(X_pca) 

plt.figure(figsize=(15,9))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c =y)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
