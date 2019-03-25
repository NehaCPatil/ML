# Write a Python program to plot quantities which have an x and y position.
import matplotlib.pyplot as plt
from week4.Matplotlib.Utility.utility import UtilityClass


class plot_quantity:
    def __init__(self):

        # creates utility class object
        self.utility_obj = UtilityClass()

    def draw_line_with_quantity(self):
        # accept size of points you wanna accept
        size = self.utility_obj.accept_size()

        # line 1 points
        x1 = self.utility_obj.CreateList(size)
        print(x1)

        y1 = self.utility_obj.CreateList(size)
        print(y1)

        # line 2 points
        x2 = self.utility_obj.CreateList(size)
        print(x2)

        y2 = self.utility_obj.CreateList(size)
        print(y2)

        # Set the x axis label
        plt.xlabel('x - axis')
        # Set the y axis label
        plt.ylabel('y - axis')

        # Sets a title
        plt.title(' quantities which have an x and y position ')

        # set new axes limits
        plt.axis([0, 100, 0, 100])

        # use pylab to plot x and y as red circles
        plt.plot(x1, y1, 'b*', x2, y2, 'ro')

        # shows the plot
        plt.show()


# creates class object
obj = plot_quantity()
# calling method by using class object
obj.draw_line_with_quantity()