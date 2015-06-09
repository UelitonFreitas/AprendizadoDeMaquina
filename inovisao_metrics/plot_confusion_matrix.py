import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='', cmap=plt.cm.Blues, class_labels = None):
    """
        Plot a confusion matrix.
        
        Args:
            cm: confusion matrix (mat format).
            cmap: mao to fill the image.
            class_labels: class labels (list of names)
            parameters: grid parameteres. Must to be a list
                ex:
                    For a SVC classifier
                    {'C' : [1,2,3,4,5]}
    
    """
    #Class index on the matrix
    index = [i for i in range(0,len(class_labels))]
    
    #Class name with biger name len.
    width = max( len(cl) for cl in class_labels)
    #Format coluns to display scores.
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 2s' for _ in enumerate(class_labels)])
    fmt += '\n'
    headers = [""] + index
    report = fmt % tuple(headers)
    report += '\n'
    
    for i,row in enumerate(cm):
        values = [class_labels[i]]
        for value in row:
            values += ["{0:2d}".format(value)]
        report += fmt % tuple(values)
    report += '\n'
    
    print report
    #clean plt
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=90)
    plt.yticks(tick_marks, class_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'-confusion-matrix.pdf')