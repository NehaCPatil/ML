import matplotlib.pyplot as plt
from week4.Matplotlib.Utility.utility import UtilityClass


class Set_Get_Axis_Values:

    # line 1 points
    utility_obj = UtilityClass()

    def draw_line(self):

        # line 1 points
        x1 = int(input("how many values do u wanna insert in x-axis for line1"))
        x1_list = self.utility_obj.CreateList(x1)
        print(x1_list)

        y1 = int(input("how many values do u wanna insert in y-axis for line1"))
        y1_list = self.utility_obj.CreateList(y1)
        print(y1_list)

        # plotting the line 1 points
        plt.plot(x1_list, y1_list, label="line 1")

        # Set the x axis label
        plt.xlabel('x - axis')
        # Set the y axis label
        plt.ylabel('y - axis')

        # Sets a title
        plt.title('Two or more lines on same plot with suitable legends ')

        # returns current axis values
        print(plt.axis())

        # accepting values to set new axis values
        print("set new axis limit")

        x_min = int(input("x_min val"))
        x_max = int(input("x_max val"))
        y_min = int(input("y_min val"))
        y_max = int(input("y_max val"))

        # sets new axis values
        plt.axis([x_min, x_max, y_min, y_max])

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()


# creates class object
obj = Set_Get_Axis_Values()

# calling method by using class object
obj.draw_line()

