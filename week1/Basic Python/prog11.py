"""Python program to check whether a file exists."""

import os.path
# Open function to open the file "abc.txt"
open('abc.txt')
# isfile() : To check if the passed argument is valid file path
print(os.path.isfile('abc.txt'))
