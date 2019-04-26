"""Python program to sort files by date."""

# import glob to use glob() and related functions
import glob
# import os to use os related functions
import os
# glob.glob() to match specified pattern according to rules
# asterisk is used to match zero or more characters
files = glob.glob("*.py")
# Sorting list of files in ascending
files.sort(key=os.path.getmtime)
# use os.path.join(), which joins paths using the correct path separator on the operating system
print("\n".join(files))
