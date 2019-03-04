# Write a Python program to concatenate all elements in a list into a string and return it.

def convert(s):
    # initialization of string to ""
    new = ""

    # traverse in the string
    for x in s:
        new += x

    # return string
    return new


s = ['p', 'y', 't', 'h', 'o', 'n']
print(convert(s))
