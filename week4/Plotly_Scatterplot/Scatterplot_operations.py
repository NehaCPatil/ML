"""1. Write a program to draw a scatter plot for random 1000 x and y coordinates
 2. Write a program to draw line and scatter plots for random 100 x and y coordinates
 3. Write a program to draw a scatter plot for random 500 x and y coordinates and style it
 4. Write a program to draw a scatter plot for a given dataset and show datalabels on hover
"""

from week4.Plotly_Scatterplot.Utility_plotly_ScatterPlot.Util import Utilclass

import re


class Matplotlib:

    # class constructor
    def __init__(self):
        self.obj1 = Utilclass()

    def calling(self):
        while True:
            try:
                print()
                print("1. Draw a scatter plot for random 1000 x and y coordinates ""\n"
                      "2. Draw line and scatter plots for random 100 x and y coordinates ""\n"
                      "3. Draw a scatter plot for random 500 x and y coordinates and style it""\n"
                      "4. Draw a scatter plot for a given data set and show data labels on hover""\n"

                      "5. Exit")
                ch = input("Enter choice:")
                choice = int(ch)
                if ch.isdigit():
                    if choice == 1:
                        print("\n Enter how many random values you want for scatter plot, on x and y axis")
                        x_axis = int(input("X axis values"))
                        y_axis = int(input("Y axis values"))
                        # both axis values should be same
                        if x_axis == y_axis:
                            self.obj1.draw_scatter_plot(x_axis, y_axis)
                        else:
                            print("Please enter same values, for both axis")

                        print("_______________________________________________________________________________")

                    elif choice == 2:
                        x_y_axis = int(input("Enter random value for  x and y axis, to draw Scatter line"))
                        self.obj1.draw_line(x_y_axis)

                        print("_______________________________________________________________________________")

                    elif choice == 3:

                        x_y_axis = int(input("Enter random value for  x and y axis, to draw Scatter line"))
                        self.obj1.draw_sctter_plot_style_it(x_y_axis)

                        print("_______________________________________________________________________________")

                    elif choice == 4:
                        """Hover effect is simply a change (of color, size, shape, image etc.) of some element 
                        while you put a mouse arrow over it."""
                        self.obj1.csv_data_set()
                        print("_______________________________________________________________________________")

                    elif choice == 5:
                        exit()
                        print("_______________________________________________________________________________")

                    else:
                        print("Plz enter valid choice: ")

                    
                    acc = str(input("IF you want to continue: type yes "))
                    if re.match(acc, 'y'):
                        continue
                    elif re.match(acc, 'yes'):
                        continue
                    elif re.match(acc, 'n'):
                        break
                    elif re.match(acc, 'no'):
                        break
                    else:
                        print("Give proper input")
                        continue

                else:
                    raise ValueError
            except ValueError as e:
                print("\nInvalid Input", e)


# class obj created
obj = Matplotlib()
obj.calling()
