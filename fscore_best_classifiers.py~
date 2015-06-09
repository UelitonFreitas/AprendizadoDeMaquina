import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os
import sys

import ReadArff.Arff as arff
from inovisao_metrics import classificarion_report_by_class as im
from inovisao_metrics import gridsearch_report_by_class as gr


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

        classes_names = dataset.get_class_names()

        # Scale feature data using MinMaxSclaer.
        # The features values are normalized with values 0 and 1.
        mm_scaler = StandardScaler()
        X_scaled = mm_scaler.fit_transform(X)

        n_jobs = -1

        # Index of classes labels.
        classes_index = dataset.get_class_names()

        # Cross validation.
        cv = KFold(len(y), shuffle=True, n_folds=10)
        cv = StratifiedKFold(y, n_folds=10)
        # C values for SVC parameters variance.
        c_values = np.logspace(-5, 15, 1000, base=2)
        g_values = np.logspace(-15, 3, 1000, base=2)
        # Parameters for grid.
        parameters = {'C': c_values, 'gamma': g_values, 'kernel': ['linear', 'rbf']}
        classifier = SVC(cache_size = 1000)

        # Macro f-measure
        my_score = make_scorer(f1_score, averange='macro')

        print "\n----------------------------------------------------"
        print "Finding best C for SVC."

        classifier = im.classificarion_report_by_class(classifier, X_scaled, y,
                                                       cv=cv,
                                                       classes_names=classes_names,
                                                       parameters=parameters,
                                                       title='SVC',
                                                       jobs=n_jobs)

        print "\n----------------------------------------------------"
        print "Finding best K for KNN."
        # K values
        


if __name__ == '__main__':
    main(sys.argv)

