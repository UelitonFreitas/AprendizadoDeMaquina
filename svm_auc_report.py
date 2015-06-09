import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from inovisao_metrics.plot_confusion_matrix import plot_confusion_matrix as pcm

import os
import sys

import ReadArff.Arff as arff

def roc_by_class_fold(classifier, X, y,class_label = None):
    # Run classifier with cross-validation and plot ROC curves
    
    cv = StratifiedKFold(y, n_folds=10)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    matrix_predicted = []
    matrix_true = []
    
    for i, (train, test) in enumerate(cv):
        classifier.fit(X[train], y[train])
        probas_ = classifier.predict_proba(X[test])
        
        matrix_predicted.extend(classifier.predict(X[test]))
        matrix_true.extend(y[test])
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
    
    #pcm(confusion_matrix(matrix_true, matrix_predicted),class_labels = ['All',class_label])
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    return (mean_fpr, mean_tpr, mean_auc)

def roc_by_class(classifier, X, y, classes_index, classes_names):
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    y_binaryze = label_binarize(y, classes_index)
    
    width = max( len(cl) for cl in classes_names)
    #Format coluns to display scores.
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 2s'])
    fmt += '\n'
    headers = [""] + ["AUC"]
    report = fmt % tuple(headers)
    report += '\n'
    
    for class_index in classes_index:
        print "Runing %s class vc All...." % classes_names[class_index]
        fpr, tpr, class_auc = roc_by_class_fold(classifier, X,
                                                y_binaryze[:,class_index], classes_names[class_index])
        
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        cn = classes_names[class_index]
        
        plt.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' %
                 (cn, class_auc))
        
        value = [cn]
        value += ["{0:2f}".format(class_auc)]
        report += fmt % tuple(value)
    report += '\n'
    
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(classes_index)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    value = ["Avg/total"]
    value += ["{0:2f}".format(mean_auc)]
    report += fmt % tuple(value)
    print report
    
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.show()
    
def main(argv):
    """Main do programa"""
    
    if len(argv) < 2:
        print "Please enter the dataset name"
    
    else:
        
        #Le um arquivo arff
        af = arff.Arff()
        file_name = str(argv[1])
        
        #Carrega um arquivo arff em um da
        dataset = af.load_arff(file_name)
        
        X = dataset.data
        y = dataset.target
        mm_scaler = MinMaxScaler()
        X_scaled =  mm_scaler.fit_transform(X)
        
        classes_index = dataset.get_class_map().values
        
        #classifier = svm.SVC(C=0.145816129947, gamma=0.00321875999118, kernel='linear', probability=True, cache_size = 4000)

        classifier = KNeighborsClassifier(n_neighbors=1,weights='uniform')
        
        #classifier = DecisionTreeClassifier(criterion='gini',splitter='best');

        #classifier = svm.SVC(C = 1635.68097512,gamma= 0.0471890060599, kernel='rbf', probability=True, cache_size = 1000)

        print classifier
        print
        
        roc_by_class(classifier, X_scaled, y, classes_index,
                     dataset.get_class_names())
        

if __name__ == '__main__':
    main(sys.argv)

