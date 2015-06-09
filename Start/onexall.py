from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
print(OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X))