import csv
import numpy as np
from sklearn import svm
from sklearn.preprocessing import RobustScaler

    
def NormalizeAllData( AllFeatureVectors ):
    #Normalization of all the feature vectors
    #Assume each row is a feature vector
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
    transformer = RobustScaler().fit( AllFeatureVectors )
    normalized_vectors = transformer.transform( AllFeatureVectors )
    return transformer, normalized_vectors

# ******************************************************
def TrainSVMModel(kernel): #kernel 'linear' 'poly' 'rbf'
    # Prepare all the data
    AllData = []
    AllLabel = []
    Realtest = []
    NumOfPostiveSamples = 0
    NumOfNegativeSamples = 0
    #Prepare negative data
    with open('Sit2dNew.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            AllData.append( row )
            AllLabel.append(0)  #Class 01 case with label 1
            line_count += 1
        print(f'Add {line_count} records as Negative samples. (Label 1) (Sit)')
        NumOfNegativeSamples = line_count
		
	#Prepare negative data
    with open('Stand2dNew.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            AllData.append( row )
            AllLabel.append(0)  #Class 02 case with label 0
            line_count += 1
        print(f'Add {line_count} records as Negative samples. (Label 0) (Stand)')
        NumOfNegativeSamples = line_count
    #Prepare positive data
    with open('Fall2dNew.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            AllData.append( row )
            AllLabel.append(1)  #Class 02 case with label 0
            line_count += 1
        print(f'Add {line_count} records as Positive samples. (Label 0) (Fall)')
        NumOfPostiveSamples = line_count

    with open('123.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            Realtest.append( row )
            line_count += 1
        print(f'Add {line_count} records as Positive samples. (Label 0) (Fall)')

        
    print( f'NumOfPostiveSamples = {NumOfPostiveSamples} \nNumOfNegativeSamples = {NumOfNegativeSamples}' )
    # ***************************
    # Normalize all the feature vectors
    AllLabel = np.array(AllLabel)
    AllData = np.array(AllData)
    Realtest = np.array(Realtest)
    
    transformer, AllFeatureVectors_normalized = NormalizeAllData( AllData )
    print (f'Feature vectors before normalization')
    for i in range(3):
        print ( AllData[i] )
    print (f'Feature vectors after normalization')
    for i in range(3):    
        print ( AllFeatureVectors_normalized[i] )
    # ***************************
    #Divide the data into two groups: TrainingData and TestingData
    trainingDataPorportion = 0.7
    
    AllDataSize = len( AllFeatureVectors_normalized )
    print(f'AllDataSize = {AllDataSize}')
    if len( AllLabel ) != AllDataSize:
        print( f'AllLabel length != AllDataSize #######################')
        raise SystemExit(f'Problem')
    
    #Make a list of randomized index
    np.random.seed(0)
    randomizedIndex = np.random.permutation(AllDataSize) #Generate a list of random number. All the numbers are from 0 to AllDataSize - 1
    #print(randomizedIndex)
    
    trainingDataSize = (int)(AllDataSize * trainingDataPorportion)
    
    TrainingData = []
    TestingData = []
    TrainingDataLabel = []
    TestingDataLabel = []
    
    #Randomly sample the AllData set to get the set of training data
    TrainingDataPosSamplesSize = 0
    TrainingDataNegSamplesSize = 0
    i = 0
    while i < trainingDataSize:
        TrainingData.append( AllFeatureVectors_normalized[randomizedIndex[i]] )
        TrainingDataLabel.append( AllLabel[randomizedIndex[i]] )
        if AllLabel[randomizedIndex[i]] == 1:
            TrainingDataPosSamplesSize += 1
        else:
            TrainingDataNegSamplesSize += 1
        i += 1
    
    TrainingDataSize = len(TrainingData)
    if TrainingDataSize != len(TrainingDataLabel):
        print( f'len(TrainingData) != len(TrainingDataLabel) #######################')
        raise SystemExit(f'Problem')
    
    print(f'Training data size = {TrainingDataSize}')
    print(f'Training data positive samples size = {TrainingDataPosSamplesSize}')
    print(f'Training data negative samples size = {TrainingDataNegSamplesSize}')
    
    #Randomly sample the AllData set to get the set of testing data (No overlapping between TrainingData and TestingData)
    TestingDataPosSamplesSize = 0
    TestingDataNegSamplesSize = 0
    i = trainingDataSize
    while i < AllDataSize:
        TestingData.append( AllFeatureVectors_normalized[randomizedIndex[i]] )
        TestingDataLabel.append( AllLabel[randomizedIndex[i]] )
        if AllLabel[randomizedIndex[i]] == 1:
            TestingDataPosSamplesSize += 1
        else:
            TestingDataNegSamplesSize += 1
        i += 1
    
    TestingDataSize = len(TestingData)
    if TestingDataSize != len(TestingDataLabel):
        print( f'len(TestingData) != len(TestingDataLabel) #######################')
        raise SystemExit(f'len(TestingData) != len(TestingDataLabel)')
    
    print(f'Testing data size = {TestingDataSize}')
    print(f'Testing data positive samples size = {TestingDataPosSamplesSize}')
    print(f'Testing data negative samples size = {TestingDataNegSamplesSize}')
    
    # ***************************************************************************
    # Train the SVM model using traing data set
    # fit the model - training
    if kernel == 'linear':
        SVM = svm.SVC(kernel='linear')
        print(f'Kernel = linear  $$$$$$$$$$$$$$$$$$$$$')
    elif kernel == 'rbf':
        SVM = svm.SVC(kernel='rbf', gamma=10)
        print(f'Kernel = rbf  $$$$$$$$$$$$$$$$$$$$$')
    elif kernel == 'poly':
        SVM = svm.SVC(kernel='poly', gamma=10)
        print(f'Kernel = poly  $$$$$$$$$$$$$$$$$$$$$')
    else:
        print(f'Invalid input kernel #########################')
        raise SystemExit(f'Invalid input kernel ')
        #sys.exit

    SVM.fit(TrainingData, TrainingDataLabel)
    TrainingData = np.array(TrainingData)
    TrainingDataLabel = np.array(TrainingDataLabel)
    print(TrainingData.shape)
    print(TrainingDataLabel.shape)
    # ******************************
    # This part can be ignored
    # Test the accuracy of the trained model using testing data
    print (f'********************')
    print (f'Testing data set classification results')
    CalculateAccuracyForTheTrainedModel( SVM, TestingData, TestingDataLabel )
    print (f'********************')
    print (f'Training data set classification results')
    CalculateAccuracyForTheTrainedModel( SVM, TrainingData, TrainingDataLabel )
    # *********************************
    #Return the trained model
    return SVM, transformer , Realtest , AllData, AllLabel

def CalculateAccuracyForTheTrainedModel( SVM, FeatureVectors_normalized, Labels ):
    # ***************************************************************************
    # Test the accuracy of the trained model using testing data
    Result = SVM.predict(FeatureVectors_normalized)  #Do prediction for each element in TestingData
    TruePositiveCount = 0
    FalsePositiveCount = 0
    TrueNegativeCount = 0
    FalseNegativeCount = 0
    
    numberOfVectors = len(FeatureVectors_normalized)
    i = 0
    while i < numberOfVectors:
        if Result[ i ] == Labels[ i ]: #True case
            if Result[ i ] == 1: #Positive case
                TruePositiveCount += 1
            else:
                TrueNegativeCount += 1 #Negative case
        else:  #False case
            if Result[ i ] == 1: #Positive case
                FalsePositiveCount += 1
            else:
                FalseNegativeCount += 1  #Negative case
        i += 1
    
    print(f'True Positive Count = {TruePositiveCount}')
    print(f'True Negative Count = {TrueNegativeCount}')
    print(f'False Positive Count = {FalsePositiveCount}')
    print(f'False Negative Count = {FalseNegativeCount}')
    
    Precision = TruePositiveCount / ( TruePositiveCount + FalsePositiveCount)
    Recall = TruePositiveCount / ( TruePositiveCount + FalseNegativeCount )
    
    print(f'Number of testing data = {numberOfVectors}')
    print(f'Precision = {Precision:0.3f}')
    #print('Precision = {:0.3f}'.format(Precision))
    print(f'Recall = {Recall:0.3f}')
    #print('Recall = {:0.3f}'.format(Recall))
    

def svmClassifyFeatureVector(SVM, transformer, featureVector):
    normalized_featureVector = transformer.transform( featureVector )
    #normalized_featureVector = transformer.transform( [featureVector] )
    # print (f'Feature vector before normalization = {featureVector}')
    # print (f'Feature vector after normalization = {normalized_featureVector[0]}')
    Result = SVM.predict( normalized_featureVector )
    return Result
# ***************************************************
if __name__ == '__main__':
    SVM, featureVectorTransformer , Realtest , AllData, AllLabel= TrainSVMModel('linear')  #kernel 'linear' 'poly' 'rbf'
    
    
    print (f'**********************************')
    #Testing feature vector
    feature_vector = AllData
    result = svmClassifyFeatureVector( SVM, featureVectorTransformer, feature_vector )
    print (f'classification result is {result}')
    
    #Testing feature vector
    feature_vector = AllData[0,:].reshape(1,38)
    result = svmClassifyFeatureVector( SVM, featureVectorTransformer, feature_vector )
    print (f'classification result is {result}')
    
    feature_vector = Realtest[3,:].reshape(1,38)
    result = svmClassifyFeatureVector( SVM, featureVectorTransformer, feature_vector )
    print (f'classification result is {result}')
    
