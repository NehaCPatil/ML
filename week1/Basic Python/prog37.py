# Python program to empty a variable without destroying it.

a = "aa"
b = 1
c = {"b": 2}
d = [3, "c"]
e = (4, 5)
letters = [a, b, c, d, e]
# type() method returns class type of the argument(object) passed as parameter
print(type(a)())
print(type(b)())
print(type(c)())
print(type(d)())
print(type(e)())
print(type(letters)())

