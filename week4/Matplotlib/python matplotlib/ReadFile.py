# 2  Write a Python program to draw a line using given axis values taken from a text file, with suitable label in
#  the x axis, y axis and a title.
import matplotlib.pyplot as plt


# class

class Readfile:

    # function for read file
    def read_text_file(self):

        # open a file
        with open("test.txt") as f:

            # reads a file
            data = f.read()

        # split a data with new line
        data = data.split('\n')

        x = [row.split(' ')[0] for row in data]
        y = [row.split(' ')[0] for row in data]

        # plotting the line points
        plt.plot(x, y)

        # Set the x axis label of the current axis.
        plt.xlabel('x - axis')

        # Set the y axis label of the current axis.
        plt.ylabel('y - axis')

        # Set a title
        plt.title('Sample graph!')

        # Display a figure.
        plt.show()


# creates class object
obj = Readfile()

# calling method by using class object
obj.read_text_file()
