""" Python program which accepts a sequence of comma-separated numbers from user
and generate a list and a tuple with those numbers."""


# getting user input
values = input("input with commas")
# split () breakup a string and add the data to a string array using a defined separator.
list = values.split()
tuple = tuple(list)
# print list
print("List:", list)
# prints the tuple
print("Tuple:", tuple)