# Python program to retrieve file properties.

from os import path
import time
# return current file path
print('File         :', __file__)
# return access time of file
print('Access time  :', time.ctime(path.getatime(__file__)))
# return Modified time of file
print('Modified time:', time.ctime(path.getmtime(__file__)))
# return the system’s ctime which, on some systems (like Unix) is the time of the last metadata change
print('Change time  :', time.ctime(path.getctime(__file__)))
# return size of file
print('Size         :', path.getsize(__file__))
# return abs path of dir
print('the current directory:', path.abspath(__file__))
# return dir name
print('the current name:', path.dirname(__file__))
# Return the base name of pathname path
print('base name:', path.basename(__file__))
print(' directory exit:', path.exists(__file__))
# Return True if path is an existing regular file
print('is file:', path.isfile(__file__))
# Return True if path is an existing regular directory
print('is directory:', path.isdir(__file__))



