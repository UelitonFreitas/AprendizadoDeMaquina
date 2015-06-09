#!/usr/bin/python
# Filename: inovisao_metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

from sklearn.cross_validation import KFold

from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import plot_confusion_matrix as pcm
    
def classificarion_report_by_class(classifier, X, y,
                                   cv = None,
                                   score = 'f1',
                                   classes_names = None,
                                   parameters = None,
                                   title = 'classifier',
                                   jobs = -1):
    """Dada um dataset, encontra os melhores parametros do classificador
    baseado na variacao dos parametros passado para a funcao utilizando
    o RandomizedSearchCV com todo o dataset. Em seguida o classificador
    e treinado utilizando validacao
    cruzada e testado. Para cada classe do probleam e apresentado a media
    da acuracia revocacao e mendida f de cada fold.
    Um grafico da matrix de confusao tambem e mostrado.
    
    Args:
        classifer: a classifier.
        X: features list.
        y: class list.
        cv: a cross validation class.
        class_names: a list containing the names of classes.
        
    Example:
        Best estimator:
        
        SVC(C=865.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
          gamma=0.0, kernel='linear', max_iter=-1, probability=False,
          random_state=None, shrinking=True, tol=0.001, verbose=False)
                                    precision     recal f-measure   support
        Report:
        acara-bandeira-marmorizado       0.16      0.16      0.15         8
                       acara-disco       0.19      0.26      0.22         8
                    barbus-sumatra       0.18      0.20      0.17         8
                       barlus-ouro       0.16      0.19      0.17         8
                       carpa-media       0.09      0.12      0.10         8
                           cascudo       0.09      0.08      0.08         8
                           dourado       0.24      0.30      0.25         8
                           kinguio       0.48      0.46      0.46         8
             kinguio-cometa-calico       0.13      0.13      0.13         8
                   kinguio-korraco       0.10      0.11      0.10         8
                       mato-grosso       0.21      0.21      0.18         8
                   molinesia-preta       0.15      0.16      0.15         8
                             oscar       0.17      0.19      0.17         8
                      oscar-albino       0.21      0.38      0.23         8
                              pacu       0.28      0.26      0.26         8
                       paulistinha       0.25      0.07      0.11         8
                  piau-tres-pintas       0.26      0.26      0.25         8
                     platy-laranja       0.78      0.66      0.68         8
                        telescopio       0.31      0.28      0.28         8
                       tetra-negro       0.14      0.10      0.11         8
                       tricogaster       0.17      0.14      0.15         8
                          tucunare       0.11      0.06      0.07         8
                                                                            
                       avg / total       0.22      0.22      0.20       8.0
                       
            Matrix de confusao
                                    Predicted class
                [[10  4  0  0  3  5  2  2  2  4  0  0  1  0  0  3  0  0  3  0  0  1]
            T   [ 0 10  1  0  3  1  0  0  2  4  1  0  2  1  4  1  2  0  3  0  4  1]
            r   [ 2  3  3  1  2  1  0  0  2  8  2  1  3  1  1  3  3  0  1  2  1  0]
            u   [ 3  1  2  8  3  0  1  2  1  0  5  3  1  2  2  0  1  2  0  2  1  0]
            e   [ 3  3  3  0  2  1  3  1  4  5  2  0  3  1  0  1  1  0  3  0  3  1]
                [ 5  3  3  0  1  3  3  2  1  1  0  1  3  3  1  1  7  0  0  0  1  1]
            c   [ 1  5  1  3  1  4  5  0  1  4  1  0  2  2  1  1  3  0  1  0  1  3]
            l   [ 7  1  1  1  1  1  0 14  1  0  1  2  1  0  0  1  0  5  0  2  0  1]
            a   [ 1  1  4  2  5  4  2  0  2  3  2  1  3  1  1  1  1  0  2  1  3  0]
            s   [ 1  6  5  1  4  2  1  1  4  4  1  3  1  1  0  0  0  0  2  2  1  0]
            s   [ 0  3  1  4  1  2  2  1  0  3  9  3  0  1  1  1  1  0  1  3  2  1]
                [ 2  2  2  3  4  1  0  1  1  3  6  3  4  1  0  1  1  0  1  3  1  0]
                [ 2  2  2  3  7  3  1  1  2  1  1  0  6  3  1  2  0  0  2  0  0  1]
                [ 1  5  2  2  0  0  0  0  1  2  1  2  1 10  2  1  3  0  1  2  1  3]
                [ 3  0  2  2  0  3  5  1  1  2  0  1  3  1  4  2  5  0  1  0  3  1]
                [ 6  4  0  6  1  3  8  1  0  0  1  2  0  0  0  0  3  1  2  1  1  0]
                [ 1  0  2  0  0  1  2  0  0  1  2  2  0  3  5  0  9  0  1  0  7  4]
                [ 0  0  0  3  0  0  1  5  0  1  1  1  0  1  0  1  0 25  0  1  0  0]
                [ 2  2  0  0  1  1  0  2  3  1  2  2  2  5  0  3  1  0 10  1  2  0]
                [ 3  0  1  5  1  0  0  4  3  1  4  2  2  0  0  3  0  4  2  3  1  1]
                [ 0  2  1  1  2  1  1  0  0  1  0  1  2  2  3  3  5  0  2  1  8  4]
                [ 1  2  2  1  0  1  3  0  0  2  2  2  1  5  2  1  8  0  1  0  4  2]]
            
            
    """
        
    #number of classes
    number_of_classes = len(classes_names)
    
    print "\nRunning RandomizedSearchCV...."
    grid = RandomizedSearchCV(classifier, parameters, cv = cv, scoring = score, n_jobs = jobs)
    grid.fit(X, y)
    print "Done!!"
    
    print ("\nBest estimator found:")
    print (grid.best_estimator_)
    
    #String utilized on report.
    last_line_heading = 'avg / total'
    
    #headers of report
    headers = ["precision", "recal", "f-measure", "support"]
    
    #Class name with bigger size
    width = max( len(cn) for cn in classes_names)
    #Compare if the bigger class name e bigger than last line
    width = max(width, len(last_line_heading))
    
    #Format coluns to display scores.
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    
    #Scores averange for each class.
    avg_by_class = {}
    
    #Confusion matrix
    matrix_predicted = []
    matrix_true = []
    
    #Initialize each score averange with 0.
    for i in range(len(classes_names)):
        avg_by_class[i] = {'precision': 0,
                           'recall': 0,
                           'f-measure': 0,
                           'support': 0}
    
    print "\nRunning cross validation..."
    #Cross validation folders.
    for i, (train, test) in enumerate(cv):
        print "Running fold %d..." % i
        #Best estimator fond by grid.
        classifier = grid.best_estimator_
        
        classifier.fit(X[train], y[train])
        predicted = classifier.predict(X[test])
        
        matrix_predicted.extend(predicted)
        matrix_true.extend(y[test])
        
        #Scores fond.
        p, r, f1, s = precision_recall_fscore_support(y[test],
                                                      predicted,
                                                      average = None)
        
        #Fit class scores with each folds scores.
        for index, label in enumerate(classes_names):
            avg_by_class[index]['precision'] += p[index]
            avg_by_class[index]['recall'] += r[index]
            avg_by_class[index]['f-measure'] += f1[index]
            avg_by_class[index]['support'] += s[index]
    
    avg_precision = 0.0
    avg_recall = 0.0
    avg_fscore = 0.0
    avg_support = 0.0
    
    #Scores for each class from all folds.
    for i in range(len(classes_names)):
        #Class name
        values = [classes_names[i]]
        
        #Final class averange.
        p = avg_by_class[i]['precision'] / len(cv)
        r = avg_by_class[i]['recall'] / len(cv)
        f = avg_by_class[i]['f-measure'] / len(cv)
        s = avg_by_class[i]['support'] / len(cv)
        
        #Format string to print.
        for v in (p, r, f):
            values += ["{0:0.2f}".format(v)]
        values += ["{0}".format(s)]
        
        avg_precision += p
        avg_recall += r
        avg_fscore += f
        avg_support += s
        report += fmt % tuple(values)
    report += '\n'
    
    #Averange of classifier
    values = [last_line_heading]
    for v in (avg_precision/number_of_classes,
              avg_recall/number_of_classes,
              avg_fscore/number_of_classes):
        values += ["{0:0.2f}".format(v)]
    values += ['{0}'.format(avg_support/number_of_classes)]
    report += fmt % tuple(values)
    
    print "Report:"
    print report
    
    #Confusion matrix
    cm = confusion_matrix(matrix_true, matrix_predicted)
    
    #plot confusion matrix.
    pcm.plot_confusion_matrix(cm, title = title, class_labels = classes_names)
    
    return grid.best_estimator_
