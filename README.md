# 839 Project Stage 1: Institution and Organization Extraction from Fake News

This project follows the guidelines for AnHai's CS 839 Data Analysis course for project stage 1, and contains an institution and organization extractor model (a random forrest), along with the preprocessing code, train and test data sets, and extra corpus files used for performing featurization and post processing.

## Introduction

This extractor parses a set of fake news articles, and extracts institutions or organizations. This includes political parties, universities, companies, websites, and other directly named social organizations. It does NOT include person names, locations, devices or named instruments. The entities in the training and test set folders have been labeled using the tags <[> and <]>. The training and test set files have words and certain kinds of punctuation split to multiple lines. If an entity spans multiple lines, the the opening tag <[> is placed before the first word of the entity (But on the same line, with no space betwee the tag and the word), and after the last word of the entity (But on the same line, with no space between the tag and the word). Below, you will find links to the different data sets we've used, to a pdf describing in further detail our process for performing the entity extraction, and instructions for running our pipeline of code.

## Links

* [Original Fake News Dataset](https://www.kaggle.com/mrisdal/fake-news) - Original set of fake news articles
* [300+ raw documents folder](data/raw_data/) - Raw documents that haven't been split to different lines
* [300+ documents folder](data/Markedup data/) - Marked up data files (Only files from 0 through 310 roughly)
* [Marked up Training Set J](data/training_set/) - 200 Marked up training files
* [Marked up Test Set I](data/test_set/) - 100 Marked up test files
* [Code folder](scripts/) - Scripts for running the pipeline and other processing
* [Compressed Directories](Archive.zip) - Zip file of this repo
* [Pdf Detailing The Project]() - Pdf describing the process of entity extraction in detail

## Running the Training Pipeline

Explain how to run the automated tests for this system

### Generating the Feature and Tuple .csv DataFrame files.

Explain what these tests test and why

```
Give an example
```

### Performing the Training CrossValidation task and generating outputs.

Explain what these tests test and why

```
Give an example
```

### Analysis of the previous step's output, full training evaluation, and false positive identification.

Explain what these tests test and why

```
Give an example
```

## Running the Testing Pipeline

Explain how to run the automated tests for this system

### Generating the Feature and Tuple .csv DataFrame files from the Training set.

Explain what these tests test and why

```
Give an example
```

### Performing extraction on the Test set.

Explain what these tests test and why

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



