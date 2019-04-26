""" Python program to get the size of an object in bytes."""

# sys module provides allows us to operate on underlying interpreter
import sys
# Creating a String
# with double Quotes
str1 = "one"
str2 = "four"
str3 = "three"
#  Printing strings size
print("Memory size of '"+str1+"' = "+str(sys.getsizeof(str1))+ " bytes")
print("Memory size of '"+str2+"' = "+str(sys.getsizeof(str2))+ " bytes")
print("Memory size of '"+str3+"' = "+str(sys.getsizeof(str3))+ " bytes")



