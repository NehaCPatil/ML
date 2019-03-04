# Python program to get the system time.

# importing "time" module for time operations
import time
print()
# using ctime() converts a time expressed in seconds since the epoch to a string representing local time
print(time.ctime())
# The time() function returns the number of seconds passed since epoch.
print(time.time())
# The gmtime() function takes the number of seconds passed since epoch as an argument and returns struct_time in UTC
print(time.gmtime())