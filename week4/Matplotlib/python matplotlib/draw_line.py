# 1 Write a Python program to draw a line using given axis values with suitable label in the x axis , y axis and a title

from matplotlib import pyplot as plt
from week4.Matplotlib.Utility.utility import UtilityClass


class draw_line:
    def __init__(self):

        # creates utility class object

        self.utility_obj = UtilityClass()

    def drawline(self):

        # line 1 points
        x_axes = int(input("how many values do u wanna insert in x-axis"))
        x_axes_list = self.utility_obj.CreateList(x_axes)
        print(x_axes_list)

        y_axes = int(input("how many values do u wanna insert in y-axis"))
        y_axes_list = self.utility_obj.CreateList(y_axes)
        print(y_axes_list)

        # Sets a title
        plt.title("Mat plot lib Title")

        # Set the x axis label
        plt.xlabel("x-axis")

        # Set the y axis label
        plt.ylabel("y-axis")

        # plotting the line points
        plt.plot(x_axes_list, y_axes_list)

        # Show the figure
        plt.show()


# creates class object
obj = draw_line()
# calling method by using class object
obj.drawline()