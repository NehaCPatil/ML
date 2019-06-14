"""Python program to list all files in a directory in Python."""

from os import listdir
from os.path import isfile, join
# isfile() : To check if the passed argument is valid file path
# listdir(): To check if passed argument is valid directory path
files_list = [f for f in listdir('/home/admin1') if isfile(join('/home/admin1', f))]
# print file list
print(files_list)
