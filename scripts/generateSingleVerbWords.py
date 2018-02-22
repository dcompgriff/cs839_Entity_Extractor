import re


'''
This script uses the AGID database from the SCOWL website to create
a verb lexicon for use as a feature for entity classification. It uses the
infl.txt file from the AGID database folder, and generates a verb.txt
lexicon.
'''

verbsList = []
with open('../../agid-2016.01.19/infl.txt','r') as f:
    for line in f:
        wordsList = line.strip().split()
        if wordsList[1] == 'V:'or wordsList[1] == 'V?:':
            verbsList.append(wordsList[0])
            for word in wordsList[2:]:
                if word != "|":
                    # Replace non characters in string with "".
                    word = re.sub("[^a-zA-Z]+", "", word)
                    if word != "":
                        verbsList.append(word.lower())


with open('./verbs.txt','w') as f:
    for verb in verbsList:
        f.write(verb + "\n")







