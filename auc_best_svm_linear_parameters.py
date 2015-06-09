#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

from sklearn.metrics import f1_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

import sys


import ReadArff.Arff as arff
import ReadArff.OneXAll as oxall


path = "Data/"

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
        
        classes_index = dataset.get_class_map().values
        
        y_binaryze = label_binarize(y, classes_index)
        
        c_values = range(1,1000,1)
        
        auc_scores_by_class = {}
        f1_scores_by_class = {}
        
        fmt = "{c:s}\t\t\t{AUC:0.4f}"
        
        best_c = None
        best_c_score = 0.0
        best_c_data = {}
        
        x_axis = []
        y_axis = []
        
        for c in c_values:
            
            classifier = LinearSVC(C = c)
            
            data = {}
            
            for class_index in classes_index:
                auc_score = cross_val_score(classifier, X_scaled, y_binaryze[:,class_index],
                                            cv = 10, scoring = 'roc_auc', n_jobs = -1)
                
                auc_scores_by_class[class_index] = auc_score.mean()
            
            auc_avg = 0.0
            
            #print "\n------------------------ Report -------------------------"
            print "C: %i\n" % c
            #print '{c:<40} {auc:<20}'.format(c = "Class Name", auc = "AUC")
            
            for key in auc_scores_by_class.keys():
                
                class_name = dataset.get_class_name_by_index(key)
                
                #print '{c:<40} {auc:<20}'.format(c=class_name,
                #                                 auc = auc_scores_by_class[key])
                
                auc_avg += auc_scores_by_class[key]
                
            nc = len(classes_index)
            #print "-----------------------------------------------------------"
            #print '{a1:<40}{a2:<20}'.format(a1 = "AUC - Avg: ",
            #                                     a2 = auc_avg/nc)
            
            auc_avg = auc_avg/nc
            
            x_axis.append(c)
            y_axis.append(auc_avg)
            
            if (auc_avg > best_c_score):
                best_c_score = auc_avg
                best_c = c
            
        plt.plot(x_axis, y_axis)
        plt.xlim([-0.05, c_values[-1]])
        plt.ylim([-0.05, 1])
        plt.xlabel('C values')
        plt.ylabel('AUC')
        plt.title('Variation')
        plt.legend(loc="lower right")
        plt.show()
            
        print "Best C: %.f\n Score: %.6f" % (best_c, best_c_score)
        

if __name__ == '__main__':
   main(sys.argv)
