#Already ran this script. Don't re run it unless u make a fresh copy from raw data.

import pandas as pd
import glob, os
import string
import re
proj = "G:/edu/sem2/839/Proj/1a/cs839_Entity_Extractor/"#git project location
data = proj+"data/"
readers = data+"raw_data/"
writers = data+"Markedup data/"
specials = "\'\’\";:.,<>?/\\|`~!@#$%^&*()-_=+[{]}"
#Non Unicode chars not included
punctuations = "\";.,?#$%&*!+-_(){}[]"
#no apostrophe in punctuations
os.chdir(readers)
#file = data+"raw_data/0_3563.txt"
for file in glob.glob("*.txt"):
	with open(file, 'r') as content_file:
		content = content_file.read()
		name = os.path.basename(content_file.name)
	#rules to make single words
	content = re.sub(r'([^\s])([\"\;\.\,\?\#\$\%\&\*\!\+\-\_\(\){}\[\]])', r'\1 \2',content)#make space before
	content = re.sub(r'([\"\;\.\,\?\#\$\%\&\*\!\+\-\_\(\){}\[\]])([^\s])', r'\1 \2',content)#make space after
	content = content.replace(" ", "\n")#\n\n implies paragraph beginning
	with open(writers+name, 'w') as content_file:
		content_file.write(content)