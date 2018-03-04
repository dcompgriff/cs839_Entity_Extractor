import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Hardcoded location for data files. 
proj = "../data/"
trainfile = proj+"train_featurized_instances.csv"
testfile = proj+"test_featurized_instances.csv"
trainTup = proj+"train_tuples_instances.csv"
testTup = proj+"test_tuples_instances.csv"

train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)

tuplesDF = pd.read_csv(testTup, index_col=0, encoding = "ISO-8859-1")
tuples = tuplesDF.as_matrix(tuplesDF.columns)


dictionary = []
with open(proj+'../data/dictionary.txt', 'r') as f:
    for line in f:
        dictionary.append(line.strip())
dictionary = set(dictionary)



def partialMatch(a, b):
    '''
    Given 2 tuples, this function tells if those two match partially.
    If we have marked one of them actual +, and predicted other as +, its still counted as partial match and should be seen as true +ve
    If we have marked one of them actual +, and predicted other as - even though it was marked +ve, its should continue to be treated as false negative
    '''
    limit = 5
    if a.file != b.file:
        return 0
    if abs(a.start-b.start)>5 or abs(a.end-b.end)>5:# no match
        return 0
    #entirely contained
    if (a.start <= b.start and a.end >= b.end):# a contains b
        return 1
    if (a.start >= b.start and a.end <= b.end):# b contains a
        return -1
    #overlapping 
    if a.start <= b.start and a.end <= b.end:
        return 2
    return 0



X_train = train.as_matrix(train.columns[1:-1])
Y_train = np.array(list(map(lambda item: 1 if item == '+' else -1, train[train.columns[-1]])))
Y_train = Y_train.reshape((Y_train.shape[0], 1))

X_test = test.as_matrix(test.columns[1:-1])
Y_test = np.array(list(map(lambda item: 1 if item == '+' else -1, test[test.columns[-1]])))
Y_test = Y_test.reshape((Y_test.shape[0], 1))

# Train Random Forest on entire training set
rfModel = RandomForestClassifier(min_samples_leaf=7, n_estimators=10, max_depth=15)
rfModel.fit(X_train, Y_train.ravel())
rfModel.feature_importances_


def postProcessingRules(predictions, tuples):
    for i in range(len(tuples)):
        if "-" in str(tuples.iloc[i][0]) or ":" in str(tuples.iloc[i][0])  or '"' in str(tuples.iloc[i][0])  or "&" in str(tuples.iloc[i][0]):
            predictions[i] = -1
        #if "ceo" == str(tuples[i][0]).lower() or "url" == str(tuples[i][0]).lower():
        #    predictions[i] = -1
        if str(tuples.iloc[i][0]).lower() in dictionary:
            predictions[i] = -1
    
    return predictions

# Run the trained model on test set
predictions = rfModel.predict(X_test)
# Run post processing rules.
predictions = postProcessingRules(predictions, tuplesDF)
#Calculate Precition and recall
pr = precision_score(Y_test, predictions, average='macro')
rc = recall_score(Y_test, predictions, average='macro')

print("Precision: ", pr)
print("Recall: ", rc)