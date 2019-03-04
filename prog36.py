# Python program to determine if variable is defined or not.

# If no exception occurs, the except clause is skipped and execution of the try statement is finished.
try:
    x = 1
    # if its type matches the exception named after the except keyword, the except clause is executed
except NameError:
    print("Variable is not defined....!")
else:
    print("Variable is defined.")

