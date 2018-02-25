#this is python 2
import sys
import os
import random
import shutil


def divide_dir():
    directory_name=sys.argv[1]
    filenames = os.listdir(directory_name)
    random.shuffle(filenames)

    #cut = random.randint(0, len(filenames))
    #list_1 = s[:199]
    #list_2 = s[200:]
    target_tr_dir = "../data/training_set/"
    target_test_dir = "../data/test_set/"
    #copy files from source dir to the target dirs
    for i in range(0,200):
        file_name = filenames[i]
    #for file_name in filenames:
        full_file_name = os.path.join(directory_name, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, target_tr_dir)
    for i in range(200,300):
        file_name = filenames[i]
    #for file_name in filenames:
        full_file_name = os.path.join(directory_name, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, target_test_dir)

if __name__ == '__main__':
    divide_dir()


