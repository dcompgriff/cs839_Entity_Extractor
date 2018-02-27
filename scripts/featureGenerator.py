import numpy as np
import pandas as pd
import argparse
import glob
import os
import time
import re
from multiprocessing import Pool


'''
****************************************************************
GLOBAL VARIABLES
****************************************************************
'''
MAX_ENTITY_LENGTH = 20
MAX_ENTITY_WORD_LENGTH = 8
NUM_FEATURES = 19
globalVerbSet = set()
with open('../data/verbs.txt', 'r') as f:
    for line in f:
        globalVerbSet.add(line.strip())

instituteKeywords = re.compile(r'\b(Inc|Incorporation|Corp|Corporation|Institute|\
University|School|College|Department|Org|Organization|Times|Committee|Foundation|Party|Agency|Council|News)\b', re.I)
badKeywords = re.compile(r'\b(The|in|as|an|III)\b', re.I)
#allow . ' ` and " inside entity words. As these are there while marking up
badpunc = re.compile(r'\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\=|\{|\}|\[|\]|\;|\:|\-|\<|\>|\,|\?|\/|\\')


globalCount = 0

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
    global globalCount
    # Create initial pandas dataframe for data objects.
    # rawString: as read form the file after removing entity markers
    # string: after stripping punctuations from inside rawString
    # wordCount: number of words in 'string' field
    # start, end: index in file 
    # class: class label if marked entity
    #tupleDF = pd.DataFrame(columns=['rawString', 'file', 'start', 'end', 'string', 'wordCount' 'class'])
    # Create native python list for appending to, which is faster than pandas DF append or concat.
    tupleList = []
    reg = re.compile(r'[a-zA-Z0-9_\’\']+')# use to strip inner punctuations, except _ and \’
    tupleColumns=['rawString', 'file', 'start', 'end', 'string', 'wordCount', 'label']
    #uniques = {}#unique string to tuple
    
    for entityLength in range(1, MAX_ENTITY_LENGTH):
        for i in range(len(fileContents)-entityLength):#reversed order here to prevent i+entityLength overflow
            # For each possible entityLength, generate string from each index.
            # Strip punctuations in order to get wordCount
            # make tuples only from those whose word count is <= MAX_ENTITY_WORD_LENGTH, >=0 and unique
            try:
                tuple = ['', fileName, i, i+entityLength, '', 0, '-']
                entityList = list(map(lambda item: str(item).strip(), fileContents[i:i+entityLength]))
                # Set class to positive if '<[>' in first list word, and '<]>' in last word in list.
                if '<[>' in entityList[0].strip() and '<]>' in entityList[-1].strip():
                    # If '<[>' and '<]>' appear in any other places internally in the string, then the
                    # string isn't a single entity, and is actually two entities that have been grouped
                    # together. Ex '<[>Project Veritas<]> shows how the <[>Clinton campaign<]>'.
                    # Count the number of times left and right tags occur in the string.
                    lCount = 0#sum(map(lambda item: 1 if '<[>' in item else 0, entityList))
                    rCount = 0#sum(map(lambda item: 1 if '<]>' in item else 0, entityList))
                    for cStr in entityList:
                        if '<[>' in cStr:
                            lCount += 1
                        if '<]>' in cStr:
                            rCount += 1
                    if lCount + rCount == 2:
                        tuple[-1] = '+'
                        globalCount += 1
                    else:
                        tuple[-1] = '-'



                # Remove any entity tags from the string.
                entityList = list(map(lambda item: item.replace('<[>', ''), entityList))
                entityList = list(map(lambda item: item.replace('<]>', ''), entityList))

                # Update the rest of the tuple information.
                tuple[0] = ' '.join(entityList)#rawString


                #################################
                # PRE-PROCESSING RULES
                #################################
                if ',' in tuple[0].strip().split()[0] or ',' in tuple[0].strip().split()[-1]:
                     continue
                # #if ('.' in tuple[0].strip().split()[0] or '.' in tuple[0].strip().split()[-1]) and len(entityList):
                # #    continue
                # if ('-' in tuple[0].strip()):
                #      continue
                if ('(' in tuple[0].strip() or ')' in tuple[0].strip()):
                     continue
                # if 'as' in tuple[0].lower() or 'a' in tuple[0].lower() or 'an' in tuple[0].lower():
                #      continue
				if len(re.findall(badpunc, tuple[0]))>0:#full tuple contains any unwanted punctuations
                    continue

                #groups of only continuous alpha numeric characters. Not including '.' as a separate group.
                words = re.findall(reg, tuple[0])
                tuple[4] = ' '.join(words)# string after stripping inner punctuations
                tuple[5] = len(words)# wordCount
                if len(re.findall(badKeywords, tuple[4])):
                    continue
                if(tuple[5]>0 and tuple[5]<=MAX_ENTITY_WORD_LENGTH):#not empty or too large a phrase
                    tupleList.append(tuple)
            except IndexError:
                continue

    return pd.DataFrame(tupleList, columns=tupleColumns)
	

def F0(tuple, fileContents):
    try:
        if fileContents[tuple.start - 1].strip().lower() == 'the' or fileContents[tuple.start - 2].strip().lower() == 'the':
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F1(tuple, fileContents):
    return sum(1 for char in tuple.string if char.isupper())

def F2(tuple, fileContents):
    try:
        if fileContents[tuple.end].strip().lower() in globalVerbSet:
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F3(tuple, fileContents):
    return len(tuple.string.strip())

def F4(tuple, fileContents):
    return tuple.wordCount#len(tuple.string.strip().split())

def F5(tuple, fileContents):
    try:
        return sum(1 for char in fileContents[tuple.start - 1] if char.isupper())
    except:
        return -1

def F6(tuple, fileContents):
    try:
        if fileContents[tuple.start - 1].strip().lower() == 'on':
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F7(tuple, fileContents):
    try:
        if fileContents[tuple.start - 1].strip().lower() == 'called':
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F8(tuple, fileContents):
    try:
        if fileContents[tuple.end].strip().lower() == 'they':
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F9(tuple, fileContents):
    try:
        if "." in tuple.rawString.split()[1:-1] or "!" in tuple.rawString.split()[1:-1] or "?" in tuple.rawString.split()[1:-1]:
            return 1
        else:
            return 0
    except IndexError:
        return 0

def F10(tuple, fileContents):
    return tuple.rawString.count('.')

def F11(tuple, fileContents):
    if ',' in tuple.rawString:
        return 1
    else:
        return 0

def F12(tuple, fileContents):
    if ',' in tuple.rawString.strip().split()[0] or ',' in tuple.rawString.strip().split()[-1]:
        return 1
    else:
        return 0

def F13(tuple, fileContents):
    if '.' in tuple.rawString.strip().split()[0] or '.' in tuple.rawString.strip().split()[-1]:
        return 1
    else:
        return 0

def F14(tuple, fileContents):
    if 'as' in tuple.rawString.lower() or 'a' in tuple.rawString.lower() or 'an' in tuple.rawString.lower():
        return 1
    else:
        return 0

def F15(tuple, fileContents):
    count = 0
    for word in tuple.rawString.strip().split():
        if word[0].isupper() and word[1:] == word[1:].lower():
            count += 1
    return count / len(tuple.rawString.strip().split())

def F16(tuple, fileContents):
    try:
        if fileContents[tuple.end][0].isupper() and fileContents[tuple.end][1:] == fileContents[tuple.end][1:].lower():
            return 1
        else:
            return 0
    except:
        return 0

def F17(tuple, fileContents):
    return 1 if len(re.findall(instituteKeywords, tuple.string))>0 else 0#case ignoring search criteria

def F18(tuple, fileContents):
    try:
        return sum(1 for char in tuple.string if char.isupper())*1.0/tuple.wordCount
    except:
        return -1
'''


Feature list:
    F0: "[The]" occurs 1 or two lines before string.
    F1: Number of capitol Letters.
    F2: Verb occurs 1 or two lines after the string.
    F3: Total character length
    F4: Total number of words
    F5: Number of capitol letters before the string.
    F5: Number of capitol letters in line after this string.
    F6: "on" comes before
    F7: "called" comes before
    F8: "they" comes after
    F9: .?! comes in the middle of and entry
    F10: Number of "."s
    F11: "," is in the raw string "NOTE: This feature reliably improves precision!"
    F12: "," is in the first or last raw string position "NOTE: This feature reliably improves precision!"
    F13: "." is in the first or last raw string position.
    F14: "as", "a", "an" is in the raw string.
    F15: The faction of the number of words where only the first character is capitalized to all words.
    F16: The rawString has a Single capitalized word after it.
	F17: Contains a keyword
	F18: fraction of capital letters to wordCount

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

    allFeaturesList.append(tuplesDF['label'].tolist())
	# TODO: write to a csv file the entire matrix of examples and features. Randomize. Remove some to ensure almost even split b/w + and -

    return pd.DataFrame(np.array(allFeaturesList).T, columns=['F' + str(i) for i in range(NUM_FEATURES)] + ['label']), tuplesDF




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

    fullDF = pd.DataFrame(columns=['F' + str(i) for i in range(NUM_FEATURES)] + ['label'])
    tuplesDF = pd.DataFrame(columns=['rawString', 'file', 'start', 'end', 'string', 'wordCount', 'label'])

    # For each file, parse into tuples, then parse into features, and create a full pandas data frame object.
    print('Performing featurization...')
    for file in fileList:
        if '.txt' in file:
            with open(args.FileFolder + file, "r", encoding="ISO-8859-1") as f:
                print(file)
                fileDF, fileTuplesDF = generateFeaturesFromFile(f.readlines(), file)
                fullDF = pd.concat([fullDF, fileDF])
                tuplesDF = pd.concat([tuplesDF, fileTuplesDF])
    endTime = time.time()
    print('Done!')
    print("Total time to run: %s seconds." %str(endTime-startTime))

    # Save the entire pandas data frame object of features and classes.
    print('Saving the full dataframe...')
    fullDF.to_csv('../data/featurized_instances.csv')
    tuplesDF.to_csv('../data/tuples_instances.csv')
    print('Done!')
    print(globalCount)





if __name__ == '__main__':
	#Parse command line arguments
	parser = argparse.ArgumentParser(description="""Fake news feature generator. Generates features from files
	whos' words have been split to multiple lines. It also handles files where entities have been pre-marked.""")
	parser.add_argument('FileFolder', metavar='f', type=str)
	args = parser.parse_args()
	main(args)

