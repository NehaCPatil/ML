"""Python program to count the number occurrence of a specific character in a string."""

# Creating a in which occurrence will be checked
s = "4444444444this is python"
# counts the number of times substring occurs in the given string and returns an integer
print(s.count("4"))
# returns length of given string
print(len(s))
# lstrip is used to delete all the leading characters mentioned in its argument.
print(s.lstrip('4'))