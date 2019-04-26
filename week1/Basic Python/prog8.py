"""Python program to create a histogram from a given list of integers."""

# define function
def histogram(items):
    for n in items:
        output = ''
        times = n
        while times > 0:
            output += '*'
            times = times - 1
        print(output)


histogram([3, 6, 9, 12])
