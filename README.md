# 839 Project Stage 1: Institution and Organization Extraction from Fake News

This project follows the guidelines for AnHai's CS 839 Data Analysis course for project stage 1, and contains an institution and organization extractor model (a random forrest), along with the preprocessing code, train and test data sets, and extra corpus files used for performing featurization and post processing.

## Introduction

Tags: <[>entity<]>

This extractor parses a set of fake news articles, and extracts institutions or organizations. This includes political parties, universities, companies, websites, and other directly named social organizations. It does NOT include person names, locations, devices or named instruments. The entities in the training and test set folders have been labeled using the tags <[> and <]>. The training and test set files have words and certain kinds of punctuation split to multiple lines. If an entity spans multiple lines, the the opening tag <[> is placed before the first word of the entity (But on the same line, with no space betwee the tag and the word), and after the last word of the entity (But on the same line, with no space between the tag and the word). Below, you will find links to the different data sets we've used, to a pdf describing in further detail our process for performing the entity extraction, and instructions for running our pipeline of code.

## Links

* [Original Fake News Dataset](https://www.kaggle.com/mrisdal/fake-news) - Original set of fake news articles
* [300+ raw documents folder](data/raw_data/) - Raw documents that haven't been split to different lines
* [300+ documents folder](data/Markedup%20data/) - Marked up data files (Only files from 0 through 310 roughly)
* [Marked up Training Set I](data/training_set/) - 200 Marked up training files
* [Marked up Test Set J](data/test_set/) - 100 Marked up test files
* [Code folder](scripts/) - Scripts for running the pipeline and other processing
* [Compressed Directories](Archive.zip) - Zip file of this repo
* [Pdf Detailing The Project]() - Pdf describing the process of entity extraction in detail

## Running the Training Pipeline

Explain how to run the automated tests for this system

### Generating the featurized_instances.csv and tuples_instances.csv DataFrame files.

#### Creating new featurized_instances.csv and tuples_instances.csv files.

The featurized_instances.csv contains a dataframe with the "featurized" entity instances and tuples_instances.csv contains the original information about the enity such as the file it occured in, the actual string of the entity, and the start and end lines. The "C" below signifies to the script that the entire data frame must be created. Note that an empty string quotations "" is passed after the C argument.

```
cd scripts/
python featureGenerator.py "../data/training_set/" C ""
```
#### Creating updated featurized_instances.csv and tuples_instances.csv from the training set.

The featurization process can be very time consuming, so the script was modified to allow for updates to the existing featurized_instances.csv file. In the below script, U is passed to the script to indicate that this is an update to the existing featurized_instances.csv file. The string after "U" represents a string list
of feature functions to update, or add to the existing featurized_instances.csv file. If the feature column exists, the values are updated. If the feature column doesn't exist, the feature column is generated and added to the existing featurized_instances.csv file. Note that the featurized_instances.csv file's columns are not garunteed to remain in sorted order. The only garuntee is that the 'label' column will be the last column.

```
cd scripts/
python featureGenerator.py "../data/training_set/" U "F10 F5 F3"
```

### Performing the Training CrossValidation tasks.

#### Perform the "Battery" suite of multiple models.

The "B" option passed to the classificationSuite.py script below indicates that the battery of model fitting and testing for linear, logistic regression, decision tree, random forrest, ada-boost, and support vector machine models should be run. Tuples of the form (precision, recall) are output for each of the 5 folds to the command line, and the order of output corresponds to the order of models previously listed in this paragraph.

```
cd scripts/
python classificationSuite.py B ../data/featurized_instances.csv ../data/tuples_instances.csv
```
#### Perform the decision tree CrossValidation with 5 folds for the decision tree and random forrests model.

The "D" option passed to the script indicates that stratified 5-fold cross-validation should be performed with the decision tree and random forrest models, and that the results should be saved in dtscores.txt and rfscores.txt files that can be used with the analysis option. The script also outputs the average precision and recall, as well as their standard deviations. Note that these can be somewhat miss-leading, as the distributions (which can be viewed with the analysis option) tend to be skewed towards the high end of these metrics and aren't symetric (meaning most of the time our models do better than the average).

```
cd scripts/
python classificationSuite.py D ../data/featurized_instances.csv ../data/tuples_instances.csv
```

#### Perform analysis of the previous step's (Option "D") output, full training evaluation, and false positive identification.

The "A" option passed to the script indicates that analysis of the output from the "D" option should be performed. The script reads the dtscores.txt and rfscores.txt files, and then generates a histogram of precision scores for the decision tree (left plot) and random forrest model (right plot). The script then outputs all false positive examples from the random forrest, and outputs a final (precision, recall) tuple for a random forrest fit to the full training set, and tested on the full training set.

```
cd scripts/
python classificationSuite.py A ../data/featurized_instances.csv ../data/tuples_instances.csv
```

## Running the Testing Pipeline

#### Creating updated featurized_instances.csv and tuples_instances.csv from the test set.

The featurization process can be very time consuming, so the script was modified to allow for updates to the existing featurized_instances.csv file. In the below script, U is passed to the script to indicate that this is an update to the existing featurized_instances.csv file. The string after "U" represents a string list
of feature functions to update, or add to the existing featurized_instances.csv file. If the feature column exists, the values are updated. If the feature column doesn't exist, the feature column is generated and added to the existing featurized_instances.csv file. Note that the featurized_instances.csv file's columns are not garunteed to remain in sorted order. The only garuntee is that the 'label' column will be the last column.

```
Give an example
```

#### Generating the trained model.

Explain

```
Give an example
```

#### Evaluating the trained model on the test set.

Explain

```
Give an example
```

## About the Feature Corpus Sets Used

We used two corpus. The first was the dictionary corpus that comes standard with linux distributions and can be found in "/usr/share/dictionary/words". The second is the [AGID](http://wordlist.aspell.net/agid-readme/) word inflection corpus which was stripped down to form a verb list which is used during the featurization process.

## Built With

* [Python 3.4.3](https://www.python.org/)
* [Numpy 1.13.1](http://www.numpy.org/) - Numerical matrix lib
* [Pandas 0.21.0](https://pandas.pydata.org/) - Data manipulation lib
* [Scikit-Learn 0.19.1](http://scikit-learn.org/stable/) - Machine learning lib
* [Matplotlib 2.0.2](https://matplotlib.org/) - Data ploting and visualization lib

## Authors

* **Daniel Griffin** - [dcompgriff](https://github.com/dcompgriff)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



