import numpy as np
import pandas as pd
import argparse
import glob
import os
import time


'''
****************************************************************
GLOBAL VARIABLES
****************************************************************
'''
MAX_ENTITY_WORD_LENGTH = 8
NUM_FEATURES = 3
globalVerbSet = set()
with open('../data/verbs.txt', 'r') as f:
    for line in f:
        globalVerbSet.add(line.strip())

'''
****************************************************************
PROGRAM FUNCTION SCRIPTS
****************************************************************
'''
'''

This function accepts a list of strings, where each
string represents a single line in a text file (which has
been pre-processed with at most 1 word on each
line.

@:param fileContents List with every line of the file contents.
@:return A pandas dataframe object that has

'''
def generateStringTuples(fileContents, fileName):
    # Create initial pandas dataframe for data objects.
    tupleDF = pd.DataFrame(columns=['string', 'file', 'lineStart', 'lineEnd', 'class'])
    # Create native python list for appending to, which is faster than pandas DF append or concat.
    tupleList = []

    for i in range(len(fileContents)):
        for entityLength in range(1, MAX_ENTITY_WORD_LENGTH):
            # For each starting index in the file, generate tuples of sizes 1 through MAX_ENTITY_WORD_LENGTH.
            try:
                tuple = ['', fileName, i, i+entityLength, '-']
                entityList = list(map(lambda item: str(item).strip(), fileContents[i:i+entityLength]))
                # Set class to positive if '<[>' in first list word, and '<]>' in last word in list.
                if '<[>' in entityList[0] and '<]>' in entityList[-1]:
                    # If '<[>' and '<]>' appear in any other places internally in the string, then the
                    # string isn't a single entity, and is actually two entities that have been grouped
                    # together. Ex '<[>Project Veritas<]> shows how the <[>Clinton campaign<]>'.
                    # Count the number of times left and right tags occur in the string.
                    lCount = sum(map(lambda item: 1 if '<[>' in item else 0, entityList))
                    rCount = sum(map(lambda item: 1 if '<]>' in item else 0, entityList))
                    if lCount + rCount > 2:
                        tuple[-1] = '-'
                    else:
                        tuple[-1] = '+'

                # Remove any entity tags from the string.
                entityList = list(map(lambda item: item.replace('<[>', ''), entityList))
                entityList = list(map(lambda item: item.replace('<]>', ''), entityList))

                # Update the rest of the tuple information.
                tuple[0] = ' '.join(entityList)
                tupleList.append(tuple)
            except IndexError:
                continue

    return pd.DataFrame(tupleList, columns=['string', 'file', 'lineStart', 'lineEnd', 'label'])


def F0(tuple, fileContents):
    try:
        if fileContents[tuple.lineStart - 1].strip().lower() == 'the' or fileContents[tuple.lineStart - 2].strip().lower() == 'the':
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F1(tuple, fileContents):
    return sum(1 for char in tuple.string if char.isupper())

def F2(tuple, fileContents):
    try:
        if fileContents[tuple.lineEnd].strip().lower() in globalVerbSet:
            return 1
        else:
            return 0
    except IndexError:
        return 0






'''


Feature list:
    F0: "[Th]e" occurs 1 or two lines before string.
    F1: Number of Capital Letters.
    F2: Verb occurs 1 or two lines after the string.
    F3:
    F4:
    F5:
    F6:
    F7:

Each "tuple" object is a Pandas series with first entry tuple[0] the index, and
    all following entries the entries of each row from the string tuples dataframe.

'''
def generateFeaturesFromFile(fileContents, fileName):
    tuplesDF = generateStringTuples(fileContents, fileName)

    allFeaturesList = []
    # Call each feature generation function on each dataframe tuple.
    for i in range(0, NUM_FEATURES):
        featureList = []
        for tuple in tuplesDF.itertuples():
            featureList.append(eval('F' + str(i) +  '(tuple, fileContents)'))
        allFeaturesList.append(featureList)

    # TODO: Add code for creating pandas feature structure column by column.




'''
****************************************************************
PROGRAM RUNNING AND MANAGEMENT SCRIPTS
****************************************************************
'''

'''

For each file in the directory provided to the program, generate all of the
possible feature sets.

'''
def main(args):
    # Get sorted file list names from the given directory.
    fileList = sorted(filter(lambda item: '.txt' in str(item), os.listdir(args.FileFolder)), key=lambda item: int(item.split('_')[0]))
    startTime = time.time()
    for file in fileList[200:300]:
        if '.txt' in file:
            with open(args.FileFolder + file, "r", encoding="ISO-8859-1") as f:
                generateFeaturesFromFile(f.readlines(), file)
    endTime = time.time()
    print("Total time to run: %s seconds." %str(endTime-startTime))















if __name__ == '__main__':
	#Parse command line arguments
	parser = argparse.ArgumentParser(description="""Fake news feature generator. Generates features from files
	whos' words have been split to multiple lines. It also handles files where entities have been pre-marked.""")
	parser.add_argument('FileFolder', metavar='f', type=str)
	args = parser.parse_args()
	main(args)

