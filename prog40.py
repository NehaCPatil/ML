# Python program to extract single key-value pair of a dictionary in variables.

# create list
student = {'name': 'JOhn', 'age': 25}
# returns a list of dict's (key, value) tuple pairs
(c1, c2) = student.items()
# print single key value pair
print(c1)
print(c2)