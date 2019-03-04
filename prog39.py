# Python program to find files and skip directories of a given directory.

from os import walk
# walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up
for (root, dirs, files) in walk("/home/admin1/PycharmProjects/neha1/"):

    print(files)