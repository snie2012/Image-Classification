"""

This PCA+MLR script is designed to implement the classic method to classify the Cifar-10 data set.

We first use PCA to reduce the number of feature from 3072 to 217, then we use the multiple logistic linear
regression to do the classification task.

"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import numpy as ny
import cPickle
from sklearn import linear_model
from sklearn.decomposition import PCA


def unpickle(file):
    """
     Unpickle the py data

    @param file:  string, name of file
    @return: a batch of data in dictionary format
    """


    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def LoadData():
    """
    Load five batch data.

    @return: a list contains training data set, training data labels, testing data set, testing data label

    """

    os.chdir('/Users/linting/Desktop/sklearnTrain/cifar-10-batches-py')

    batch1 = unpickle('data_batch_1')
    batch2 = unpickle('data_batch_2')
    batch3 = unpickle('data_batch_3')
    batch4 = unpickle('data_batch_4')
    batch5 = unpickle('data_batch_5')
    testbatch = unpickle('test_batch')

    batch1_data = batch1['data'].tolist()
    batch2_data = batch2['data'].tolist()
    batch3_data = batch3['data'].tolist()
    batch4_data = batch4['data'].tolist()
    batch5_data = batch5['data'].tolist()
    test_data = testbatch['data'].tolist()

    batch1_data_labels = batch1['labels']
    batch2_data_labels = batch2['labels']
    batch3_data_labels = batch3['labels']
    batch4_data_labels = batch4['labels']
    batch5_data_labels = batch5['labels']
    test_data_labels = testbatch['labels']

    train_data_RGB = batch1_data + batch2_data + batch3_data + batch4_data + batch5_data
    train_data_labels = batch1_data_labels + batch2_data_labels + batch3_data_labels +batch4_data_labels +batch5_data_labels

    return [train_data_RGB, train_data_labels, test_data, test_data_labels]



def PCA_GetComponents():
    """
    Reduce the number of feature components using pca. Then keep the number components that has the percent
    of cumulative variance of 95%.
    :return: None

    """
    data = LoadData()

    train_data_RGB = data[0]


    pca = PCA(n_components=3072)
    pca.fit(train_data_RGB).fit_transform(train_data_RGB)
    Cumulative_variance = pca.explained_variance_ratio_
    return Cumulative_variance


def main():
    """
    Transformed the data to a new space by PCA.
    Import the tranformed data to MLR to do classfication.

    :return: float, accuracy of classification
    """

    data = LoadData()
    train_data_RGB = data[0]
    train_data_labels = data[1]
    test_data = data[2]
    test_data_labels = data[3]
    train_data_RGB = ny.array(train_data_RGB)
    test_data = ny.array(test_data)

    # Using PCA to transform traning data and test data
    pca = PCA(n_components=217)
    transformed_train_data = pca.fit_transform(train_data_RGB)
    transformed_test_data = pca.transform(test_data)

    # MLR to do classification .
    mlr = linear_model.LogisticRegression()
    print "Training data..."
    mlr.fit(transformed_train_data, train_data_labels)

    print "Predict test data...."
    pred_data_labels = mlr.predict(transformed_test_data)
    pred_data_labels = pred_data_labels.astype(int).tolist()

    confusion = confusion_matrix(test_data_labels, pred_data_labels)
    print 'Confusion matrix is:'
    print confusion

    accuracy =accuracy_score(test_data_labels, pred_data_labels)

    return accuracy


if __name__ =="__main__":
    print "Classification accuracy is %f"  % main()



