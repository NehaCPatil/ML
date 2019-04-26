""" Python program to get an absolute file path."""

# define function
def absolute_file_path(path_fname):
    #  importing "os" module for using operating system dependent functionality
    import os
    return os.path.abspath(path_fname)


# It returns the absolute path of the file/directory name passed as an argument.
#try:
# print("Absolute file path: ", absolute_file_path("a.txt"))

#except Exception as e
# Docstrings access
print(abs.__doc__)

