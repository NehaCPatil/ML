""" Python program to get the current value of the recursion limit."""

# module provides access to some variables used or maintained by the interpreter
# and to functions that interact strongly with the interpreter
import sys
#  Recursion implies it's the same thread over and over again that it's limiting.
#  printing recursion limit
print(sys.getrecursionlimit())
# set recursion limit
sys.setrecursionlimit(3000)
# printing updated recursion limit
print(sys.getrecursionlimit())