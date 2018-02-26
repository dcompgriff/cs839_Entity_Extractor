import numpy as np
import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time
import matplotlib.pyplot as plt

'''
##########################################################
GLOBAL ML MODELS
##########################################################
'''
linearModel = LinearRegression()
logisticModel = LogisticRegression()
dtModel = DecisionTreeClassifier(min_samples_leaf=7)
rfModel = RandomForestClassifier(min_samples_leaf=7, n_estimators=10, max_depth=15)
abModel = AdaBoostClassifier()  # base_estimator=SVC(C=0.1, tol=0.1, verbose=1, kernel='linear', probability=True), n_estimators=10)
# 50 for data.
svmModel = SVC(C=0.1, tol=1, verbose=1, kernel='linear')  # ( , class_weight='balanced')


'''
##########################################################
SUITE FUNCTIONS
##########################################################
'''
def runBattery(args):
    dataDF = pd.read_csv(args.PandasDataFrame)

    # Generate X, Y data sets.
    X = dataDF.as_matrix(dataDF.columns[1:-1])
    Y = np.array(list(map(lambda item: 1 if item == '+' else -1, dataDF[dataDF.columns[-1]])))
    Y = Y.reshape((Y.shape[0], 1))

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random)
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Fit each model.
        print('Training linear model...')
        linearModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training logistic model...')
        logisticModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training decision tree model...')
        dtModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training random forrests model...')
        rfModel.fit(X_train, y_train.ravel())
        print('Done!')
        print('Training svm model...')
        startTime = time.time()
        svmModel.fit(X_train, y_train.ravel())
        endTime = time.time()
        print('Done! %s' % (str(endTime - startTime)))
        print('Training adaboost model...')
        abModel.fit(X_train, y_train.ravel())
        print('Done!')

        # Evaluate each model.
        scoring = ['precision_macro', 'recall_macro']
        scores = [0 for i in range(6)]
        print('Testing linear model...')
        predictions = linearModel.predict(X_test)
        predictions = list(map(lambda item: 1 if item >= 0 else -1, predictions))
        scores[0] = (
        precision_score(y_test, predictions, average='macro'), recall_score(y_test, predictions, average='macro'))
        print('Done!')
        print('Testing logistic model...')
        predictions = linearModel.predict(X_test)
        predictions = list(map(lambda item: 1 if item >= 0.5 else -1, predictions))
        scores[1] = (
        precision_score(y_test, predictions, average='macro'), recall_score(y_test, predictions, average='macro'))
        print('Done!')
        print('Testing decision tree...')
        scores[2] = (precision_score(y_test, dtModel.predict(X_test), average='macro'),
                     recall_score(y_test, dtModel.predict(X_test), average='macro'))
        print('Done!')
        print('Testing random forrest model...')
        scores[3] = (precision_score(y_test, rfModel.predict(X_test), average='macro'),
                     recall_score(y_test, rfModel.predict(X_test), average='macro'))
        print('Done!')
        print('Testing SVM model...')
        startTime = time.time()
        scores[4] = (precision_score(y_test, svmModel.predict(X_test), average='macro'),
                     recall_score(y_test, svmModel.predict(X_test), average='macro'))
        endTime = time.time()
        print('Done! %s' % (str(endTime - startTime)))
        print('Testing adaboost model...')
        scores[5] = (precision_score(y_test, abModel.predict(X_test), average='macro'),
                     recall_score(y_test, abModel.predict(X_test), average='macro'))
        print('Done!')

        # Print results.
        print(scores)

def runDT(args):
    dataDF = pd.read_csv(args.PandasDataFrame)

    # Generate X, Y data sets.
    X = dataDF.as_matrix(dataDF.columns[1:-1])
    Y = np.array(list(map(lambda item: 1 if item == '+' else -1, dataDF[dataDF.columns[-1]])))
    Y = Y.reshape((Y.shape[0], 1))

    dtScores = []
    rfScores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random)
    fold = 0
    startTime = time.time()
    for train_index, test_index in skf.split(X, Y):
        print("Fold %s"%(str(fold)))
        fold += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Fit each model.
        #print('Training decision tree model...')
        dtModel.fit(X_train, y_train.ravel())
        #print('Done!')
        #print('Training random forrests model...')
        rfModel.fit(X_train, y_train.ravel())
        #print('Done!')

        # Evaluate each model.
        scoring = ['precision_macro', 'recall_macro']
        #print('Testing decision tree...')
        dtScores.append([precision_score(y_test, dtModel.predict(X_test), average='macro'), recall_score(y_test, dtModel.predict(X_test), average='macro')])
        #print('Done!')
        #print('Testing random forrest...')
        rfScores.append([precision_score(y_test, rfModel.predict(X_test), average='macro'), recall_score(y_test, rfModel.predict(X_test), average='macro')])
        #print('Done!')
    endTime = time.time()
    print("Time to run: %s seconds."%(str(endTime-startTime)))

    # Print results.
    dtScores = np.array(dtScores)
    rfScores = np.array(rfScores)
    print("DT: Avg (%s,%s), StdDev (%s,%s)"%(dtScores.mean(0)[0],dtScores.mean(0)[1],dtScores.std(0)[0],dtScores.std(0)[1]))
    print("RF: Avg (%s,%s), StdDev (%s,%s)" % (rfScores.mean(0)[0], rfScores.mean(0)[1], rfScores.std(0)[0], rfScores.std(0)[1]))

    with open('dtscores.txt', 'w') as f:
        for score in dtScores:
            f.write("%s,%s\n"%(score[0], score[1]))

    with open('rfscores.txt', 'w') as f:
        for score in rfScores:
            f.write("%s,%s\n"%(score[0], score[1]))

def runDTAnalysis(args, plot=True):
    dtScores = []
    rfScores = []
    with open('dtscores.txt', 'r') as f:
        for line in f:
            dtScores.append([float(line.strip().split(',')[0]), float(line.strip().split(',')[1])])

    with open('rfscores.txt', 'r') as f:
        for line in f:
            rfScores.append([float(line.strip().split(',')[0]), float(line.strip().split(',')[1])])
    dtScores = np.array(dtScores)
    rfScores = np.array(rfScores)

    # Show precision histograms if possible.
    if plot:
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(dtScores[:,0], range=(0.5,1))
        axs[1].hist(rfScores[:,0], range=(0.5,1))
        plt.show()

    # Analyze false positives.
    dataDF = pd.read_csv(args.PandasDataFrame)
    tuplesDF = pd.read_csv(args.TuplesPandasDataFrame)
    # Generate X, Y data sets.
    X = dataDF.as_matrix(dataDF.columns[1:-1])
    Y = np.array(list(map(lambda item: 1 if item == '+' else -1, dataDF[dataDF.columns[-1]])))
    Y = Y.reshape((Y.shape[0], 1))
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random)
    #fold = 0
    startTime = time.time()
    #train_index, test_index = list(skf.split(X, Y))[0]
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = Y[train_index], Y[test_index]
    dtModel.fit(X, Y.ravel())
    predictions = dtModel.predict(X)
    for i in range(len(predictions)):
        if Y[i] == -1 and predictions[i] == 1:
            # False Positive Found, so print it out.
            print(tuplesDF.iloc[i])




def main(args):
    if args.FunctionToRun == "B":
        runBattery(args)
    elif args.FunctionToRun == "D":
        runDT(args)
    elif args.FunctionToRun == "A":
        runDTAnalysis(args)
    else:
        print("Incorrect function to run specification.")







if __name__ == '__main__':
    #Parse command line arguments
    parser = argparse.ArgumentParser(description="""Entity Classifier Suite. This script trains and tests multiple classifiers
    on the pandas data frame passed to this.""")
    parser.add_argument('FunctionToRun', metavar='fr', type=str, help="B is run battery, D is run decision tree, A is run decision tree Analysis (D must be run first)")
    parser.add_argument('PandasDataFrame', metavar='f', type=str)
    parser.add_argument('TuplesPandasDataFrame', metavar='fr', type=str, help="B is run battery, D is run decision tree, A is run decision tree Analysis (D must be run first)")
    args = parser.parse_args()
    main(args)












