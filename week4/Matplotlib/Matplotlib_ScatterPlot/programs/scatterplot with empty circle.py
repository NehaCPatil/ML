from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt

from pylab import randn


# class to perform graphical representation of data using matplotlib pie chart
class ScatterPlotEmptyCircle:
    choice = 0

    def scatter_plots(self):
        print()
        print("1. Draw a scatter plot with empty circles taking a random distribution in X and Y "
              "and plotted against each other")
        print("2. Exit")
        print()
        while True:
            try:
                print()
                # accept choice from user
                self.choice = input("Enter choice : ")
                # validate choice number
                valid_choice = validate_num(self.choice)
                if valid_choice:
                    choice = int(self.choice)
                    if choice == 1:
                        x_value = get_random_distribution()
                        y_value = get_random_distribution()
                        # generate a scatter plots
                        plt.scatter(x_value, y_value, s=70, facecolors='none', edgecolors='g')
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.show()
                    elif choice == 2:
                        exit()
                    else:
                        print("Enter valid choice")
                else:
                    print("Enter only numbers")
            except Exception as e:
                print(e)


# class object
ScatterPlotEmptyCircle_object = ScatterPlotEmptyCircle()
ScatterPlotEmptyCircle_object.scatter_plots()
