""" Python program to sort three integers without using conditional statements and loops."""


# getting user input first no
x = int(input("enter first no"))
# getting user input second no
y = int(input("enter second num"))
# getting user input third no
z = int(input("enter third num"))

# compute the min of the values passed in its argument
a1 = min(x, y, z)
# compute the maximum of the values passed in its argument
a3 = max(x, y, z)
# compute the middle of the values passed in its argument
a2 = (x+y+z) - a1 - a3
# print num in sorted order
print("num in sorted order", a1, a2, a3)
