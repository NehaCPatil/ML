# 15. Write a Python program to create multiple plots.

import matplotlib.pyplot as plt
from week4.Matplotlib.Utility.utility import UtilityClass


class subplot:
    # creates utility class object
    utility_obj = UtilityClass()

    def create_subplot(self):

        # accept size of points you wanna accept
        size = self.utility_obj.accept_size()

        # line 1 points
        print("enter line 1 x axis values")
        x1 = self.utility_obj.CreateList(size)
        print(x1)
        print("enter line 1 y axis values")
        y1 = self.utility_obj.CreateList(size)
        print(y1)

        # Sets up a subplot grid that has height 2 and width 1,
        # and set the first such subplot as active.
        plt.subplot(2, 1, 1)

        # plotting the line 1 points
        plt.plot(x1, y1, label="line 1")

        # Sets a title
        plt.title('subplot1')

        print("line 2")
        # accept size of points you wanna accept for line 2
        size_l2 = self.utility_obj.accept_size()

        # line 2 points
        print("enter line 2 x axis values")
        x2 = self.utility_obj.CreateList(size_l2)
        print(x2)
        print("enter line 2 y axis values")
        y2 = self.utility_obj.CreateList(size_l2)
        print(y2)

        # Set the second subplot as active, and make the second plot.
        plt.subplot(2, 1, 2)

        # plotting the line 2 points
        plt.plot(x2, y2, label="line 2")

        # Sets a title
        plt.title('subplot2')

        # Shows the figure.
        plt.show()


# creates class object
obj = subplot()
# calling method by using class object
obj.create_subplot()
