import pandas as pd
proj = "G:/edu/sem2/839/Proj/1a/cs839_Entity_Extractor"#git project location
rn = pd.read_csv(proj+"/data/randnums.txt")
data = pd.read_csv(proj+"/data/fake.csv")
j=0
for i in range(len(rn)):
    name=proj+"/data/raw_data/"+str(j)+"_"+str(rn.Index[i])+".txt";
    f=open(name,"w")
    try:
        f.write(data.text.iloc[rn.Index[i]])
        j=j+1
    except UnicodeEncodeError:
        print(rn.Index[i])
    f.close()