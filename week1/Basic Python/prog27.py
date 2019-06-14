"""Python program to get the system time."""

# importing "time" module for time operations
import time

# using ctime() converts a time expressed in seconds since the epoch to a string representing local time
print('\n' + time.ctime() + '\n' + (time.time()))
# The time() function returns the number of seconds passed since epoch.
