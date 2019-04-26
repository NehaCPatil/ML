# Python program to add leading zeroes to a string.


s = "this is string example....wow!!!";
# zfill() method returns a copy of the string with '0' characters padded to the left
print(s.zfill(40))
print(s.zfill(70))
# ljust() returns the string left justified in a string of length width
print(s.ljust(60, '0'))