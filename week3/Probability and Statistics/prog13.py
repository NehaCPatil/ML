"""13. The table below shows the height, x, in inches and the pulse rate, y, per minute,
for 9 people. Write a program to find the correlation coefficient and interpret your result.
x ⇒ 68 72 65 70 62 75 78 64 68
y ⇒ 90 85 88 100 105 98 70 65 72"""

# Python Program to find correlation coefficient.
import math


# function that returns correlation coefficient.
def correlationCoefficient(X, Y, n):
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0

    i = 0
    while i < n:
        # sum of elements of array X.
        sum_X = sum_X + X[i]

        # sum of elements of array Y.
        sum_Y = sum_Y + Y[i]

        # sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i]

        # sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i]
        squareSum_Y = squareSum_Y + Y[i] * Y[i]

        i = i + 1

    # use formula for calculating correlation
    # coefficient.
    corr = float(n * sum_XY - sum_X * sum_Y) / float(
        math.sqrt((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)))

    return corr


# Driver function
X = [68, 72, 65, 70, 62, 75, 78, 64, 68]
Y = [90, 85, 88, 100, 105, 98, 70, 65, 72]

# Find the size of array.
n = len(X)

# Function call to correlationCoefficient.
print('{0:.6f}'.format(correlationCoefficient(X, Y, n)))


