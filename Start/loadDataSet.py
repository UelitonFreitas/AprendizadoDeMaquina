from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

clf = svm.SVC(gamma=0.001, C=100.)

p = clf.get_params()

print "Params"
print p

print(digits.data)

print "target:"
print(digits.target)

print "Image 0"
print digits.images[0]
print ""
print digits.images[1]