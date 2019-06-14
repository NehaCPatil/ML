""" Python program to print the calendar of a given month and year"""

# import module
import calendar
# yy = 2014
# mm = 11
# To ask month and year from the user
yy = int(input("Enter year: "))
mm = int(input("Enter month: "))
# display the calendar
print(calendar.month(yy, mm))

