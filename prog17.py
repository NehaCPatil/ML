# program to get execution time for a Python method.

#  importing "time" module for time operations
import time
# define function
def sum_of_n_numbers(n):
    # using time() to display time
    start_time = time.time()
    s = 0
    for i in range(1, n+1):
        s = s + i
    end_time = time.time()
    return s, end_time - start_time


n = 5
# print time in sec
print(sum_of_n_numbers(n))

