# Python function to find the maximum and minimum numbers from a sequence of numbers.

# define function for highest num
def highestNumber(l):
    myMax = l[0]
    for num in l:
        if myMax < num:
            myMax = num
    return myMax

# define function for highest num
def smallestNumber(l):
    myMin = l[0]
    for num in l:
        if myMin > num:
            myMin = num
    return myMin


# return highest num in list
print(highestNumber([77, 48, 19, 17, 93, 90]))
# return smallest num in list
print(smallestNumber([77, 48, 19, 17, 93, 90]))
