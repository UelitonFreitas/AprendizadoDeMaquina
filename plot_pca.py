import numpy as np

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os
import sys

import ReadArff.Arff as arff


def main(argv):
    """
        Encontra os melhores parametros para os classificadores SVC ,KNN
        e arvire de decissa (CART).
        Para cada classificador otimizado retorna:
            Os scores por classe.
            Matriz de confusao
    """

    if len(argv) < 2:
        print "Please enter the data set name"

    else:

        # Arff file.
        af = arff.Arff()
        file_name = str(argv[1])

        # Load a Arff File.
        dataset = af.load_arff(file_name)

        # Features.
        X = dataset.data
        # class labels.
        y = dataset.target
        y = y[0:3*40]
        classes_names = dataset.get_class_names()

        # Scale feature data using MinMaxSclaer.
        # The features values are normalized with values 0 and 1.
        mm_scaler = MinMaxScaler()
        X_scaled = mm_scaler.fit_transform(X)

        # Index of classes labels.
        classes_index = dataset.get_class_names()

        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        pca = decomposition.PCA(n_components=3)
        pca.fit(X_scaled)
        X = pca.transform(X_scaled)

        for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
            ax.text3D(X[y == label, 0].mean(),
                      X[y == label, 1].mean() + 1.5,
                      X[y == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

        x_surf = [X[:, 0].min(), X[:, 0].max(),
                  X[:, 0].min(), X[:, 0].max()]
        y_surf = [X[:, 0].max(), X[:, 0].max(),
                  X[:, 0].min(), X[:, 0].min()]
        x_surf = np.array(x_surf)
        y_surf = np.array(y_surf)
        v0 = pca.transform(pca.components_[0])
        v0 /= v0[-1]
        v1 = pca.transform(pca.components_[1])
        v1 /= v1[-1]

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        plt.show()




if __name__ == '__main__':
    main(sys.argv)