__author__ = 'linting'
from SimpleCV.Features import HueHistogramFeatureExtractor
from SimpleCV.Features import EdgeHistogramFeatureExtractor

import SimpleCV.MachineLearning.SVMClassifier as SVMClassifier
from SimpleCV import Camera, ImageSet
import random

import datetime
start_time =  datetime.datetime.now().time()
class Trainer():

    def __init__(self,classes, trainPaths):
        self.classes = classes
        self.trainPaths = trainPaths


    def getExtractors(self):
        hhfe = HueHistogramFeatureExtractor(10)
        ehfe = EdgeHistogramFeatureExtractor(10)

        return [hhfe,ehfe]

    def getClassifiers(self,extractors):
        svm = SVMClassifier(extractors)

        return [svm]

    def train(self):
        print 'feature extraction'
        self.classifiers = self.getClassifiers(self.getExtractors())
        for classifier in self.classifiers:
            print 'trainning'
            classifier.train(self.trainPaths,self.classes,verbose=False)

    def test(self,testPaths):
        for classifier in self.classifiers:
            print classifier.test(testPaths,self.classes, verbose=True)


classes = ['0','1','2','3','4','5','6','7','8','9']

def main():

    data_batch_1 = ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/data_batch_1/'+c  for c in classes ]
    data_batch_2 = ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/data_batch_2/'+c  for c in classes ]
    data_batch_3 = ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/data_batch_3/'+c  for c in classes ]
    data_batch_4 = ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/data_batch_4/'+c  for c in classes ]
    data_batch_5 = ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/data_batch_5/'+c  for c in classes ]

    trainPaths = data_batch_1 + data_batch_2 + data_batch_3 + data_batch_4 + data_batch_5
    testPaths =  ['/Users/linting/Desktop/CSC522/Project/simplecv/data/cifar-10-batches-mat/test_batch/'+c for c in classes]

    trainer = Trainer(classes,trainPaths)
    print "Start training....."
    trainer.train()

    imgs = ImageSet()
    for p in testPaths:
        imgs += ImageSet(p)
    random.shuffle(imgs)

    print "Result test"
    trainer.test(testPaths)


main()
print 'StartTime:', start_time.isoformat()
end_time = datetime.datetime.now().time()
print 'EndTime:', end_time.isoformat()

