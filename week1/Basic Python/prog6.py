"""Python program to calculate number of days between two dates."""

# import module
from datetime import date
# define function
def numOfDays(date1, date2):
    return (date2 - date1).days


date1 = date(2018, 12, 13)
date2 = date(2019, 2, 25)
# prints the num of Days
print(numOfDays(date1, date2), "days")
