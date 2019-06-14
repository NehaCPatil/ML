"""Python program to get file creation and modification date/times."""


import os.path
# importing "time" module for time operations
import time

# using getmtime() to return the  last time the file's contents were changed
print("Last modified: %s", time.ctime(os.path.getmtime("abc.txt")))

#  using ctime() converts a time expressed in seconds since the epoch to a string representing local time
#  using getctime() to return the file creation time

print("Created: %s", time.ctime(os.path.getctime("abc.txt")))

print(lname, fname)
