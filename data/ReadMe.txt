Steps:
	Get fake-news.csv from kaggle https://www.kaggle.com/mrisdal/fake-news/data
	Get randomnums.txt from git
	use data/makeFiles.py to generate txt files from the fake.csv files
	The new files will be in a "raw data" folder. Manually delete all zero-sized files. This will not affect indexing
	OR just copy the rawdata folder from git itself
	Make a copy of the "raw data" folder and call it "Markedup data"
	There are 465 files in each of these folders.
	Mitali takes 0-99 index.
	Yudhister takes 100-199
	Dan takes 200-299
	Markup the data in "Markedup data" folder to label all institution names. 
	
	Don't commit your "markedup data" folder untill we have satisfactorily marked up all the docs in it. 
