""" Write a Python program to find out the number of CPUs using."""


import multiprocessing
import os
# print num of  using  CPUs
print("Number of CPU's using :", multiprocessing.cpu_count())
print("Number of CPU's using :", os.cpu_count())