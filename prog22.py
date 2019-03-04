# Python program to get the command-line arguments
# (name of the script, the number of arguments, arguments) passed to a script.

# sys module provides allows us to operate on underlying interpreter
import sys
# collects the String arguments passed to the python script
print("This is the name/path of the script:", sys.argv[0])
# count String arguments passed to the python script
print("Number of arguments:", len(sys.argv))
# display list of args
print("Argument List:", str(sys.argv))
