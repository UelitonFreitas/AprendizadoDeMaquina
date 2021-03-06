print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
import ReadArff.Arff as arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

import sys

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring='f1', cv=10, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    print test_scores_mean
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def main(argv):
    
    """Main do programa"""
    
    if len(argv) < 2:
        print "Please enter the dataset name"
    
    else:
        af = arff.Arff()
        file_name = str(argv[1])
        dataset = af.load_arff(file_name)
        
        X = dataset.data
        y = dataset.target
        
        mm_scaler = MinMaxScaler()
        
        X_scaled =  mm_scaler.fit_transform(X)
        
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        
        title = "Learning Curves (SVM, Linear kernel, $c=52$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        
        cv = cross_validation.KFold(len(y),shuffle = True, n_folds=10)
        
        estimator = SVC(C = 725, kernel='linear')
        
        plot_learning_curve(estimator, title, X_scaled, y, (0.0, 1.01), cv=cv, n_jobs=-1)
        
        plt.show()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
        
        plot_learning_curve(estimator, title, X_train, y_train, (0.0, 1.01), cv=cv, n_jobs=-1)
        plt.show()
        plot_learning_curve(estimator, title, X_test, y_test, (0.0, 1.01), cv=cv, n_jobs=-1)
        plt.show()

if __name__ == '__main__':
    main(sys.argv)
