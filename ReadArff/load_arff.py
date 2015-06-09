__author__ = 'ueliton'

from scipy.io import arff
from sklearn.feature_extraction import DictVectorizer

from pandas import DataFrame
from pandas import Series
import numpy as np

import Dataset as ds

def load_arff(a_file=None, class_attribute=None):
    """
        Load a arff file and return a dataset.

        :args
            a_file: arrf file name.
            class_attribute: the attribute class label.

        :return
            Dataset: a data set containing all arff data.
    """

    np.set_printoptions(threshold=np.nan)

    print 'Loading Arff file %s' % (a_file)
    # data contains all values of features.
    # meta contains arff file information.
    data, meta = arff.loadarff(a_file)
    print "Load complete!!"
    print data
    # Data form to be used in scikit.
    data_frame = DataFrame(data)
    print "\nArff attributes:"
    print meta

    # All class values.
    classes = data_frame[class_attribute]

    # Unique values of class attribute.
    classes_labels = np.unique(classes)
    print("Problem classes:")

    # Mapping the values into numeric values.
    class_map = Series([x[0] for x in enumerate(classes_labels)],
                           index=classes_labels)

    # If class attribute is not numeric.
    if( meta[class_attribute][0] == "nominal" ):
        classes = classes.map(class_map)

    print (class_map)

    # Delete class attribute from data_frame.
    # The class attribute must be a vector type in Scikit-Learn.
    del data_frame[class_attribute]

    # Transform nominal attributes.
    for att in meta:

        # If attribute is nominal.
        if att != class_attribute and meta[att][0] == "nominal":
            # Use DictVectorizer to index nominal values.
            vec = DictVectorizer()

            # List of dictionaries that contains each value occurrences.
            #   ex:
            #     measurements = [
            #         {'city': 'Dubai', 'temperature': 33.},
            #         {'city': 'London', 'temperature': 12.},
            #         {'city': 'San Fransisco', 'temperature': 18.},
            #     ]
            measures = []

            # For each nominal attribute value insert into measures.
            for value in data_frame[att]:
                measures.append({att:value})

            # Array of values
            #       column  0       1       2               3
            #               Dubai   London  San Fransisco   temperature
            # vec_data = array([
            #               [  1.,   0.,   0.,  33.],
            #               [  0.,   1.,   0.,  12.],
            #               [  0.,   0.,   1.,  18.]
            #           ])
            vec_data = vec.fit_transform(measures).toarray()

            # Delete old nominal attribute.
            del data_frame[att]

            # Create a column in data_frame for each nominal value.
            for index, value_name in enumerate(vec.get_feature_names()):
                # data_frame['city=Dubay'] = [
                #     1,
                #     0,
                #     1
                # ]
                data_frame[value_name] = vec_data[:,index]

    return ds.Dataset(data,meta,data_frame,classes,class_map)
    