# Write a Python program to plot two or more lines with legends, different widths and colors.

import matplotlib.pyplot as plt
from week4.Matplotlib.Utility.utility import UtilityClass


class plot_two_lines:

    # creates utility class object
    utility_obj = UtilityClass()

    def draw_line(self):

        # line 1 points
        x1 = int(input("how many values do u wanna insert in x-axis for line1"))
        x1_list = self.utility_obj.CreateList(x1)

        print(x1_list)

        y1 = int(input("how many values do u wanna insert in y-axis for line1"))
        y1_list = self.utility_obj.CreateList(y1)
        print(y1_list)

        # line 2 points
        x2 = int(input("how many values do u wanna insert in x-axis for line2"))
        x2_list = self.utility_obj.CreateList(x2)

        print(x2_list)

        y2 = int(input("how many values do u wanna insert in y-axis for line2"))
        y2_list = self.utility_obj.CreateList(y2)
        print(y2_list)

        # Set the x axis label
        plt.xlabel('x - axis')

        # Set the y axis label
        plt.ylabel('y - axis')

        # Sets a title
        plt.title('Two or more lines on same plot with legends, different widths and colors ')

        # plotting the line 1 points with color,width,label
        plt.plot(x1_list, y1_list, color='blue', linewidth=2, label="line1-width-2")

        # plotting the line 2 points with color,width,label
        plt.plot(x2_list, y2_list, color='black', linewidth =4, label="line2-width-4")

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()


# creates class object
obj = plot_two_lines()
# calling method by using class object
obj.draw_line()

