#Already ran this script. Don't re run it unless u make a fresh copy from raw data.

import pandas as pd
import glob, os
import string
proj = "G:/edu/sem2/839/Proj/1a/cs839_Entity_Extractor/"#git project location
data = proj+"data/Markedup data/"
os.chdir(data)
for file in glob.glob("*.txt"):
    with open(file, 'r') as content_file:
        content = content_file.read()
    content = content.replace(" ", "\n")
    with open(file, 'w') as content_file:
        content_file.write(content)