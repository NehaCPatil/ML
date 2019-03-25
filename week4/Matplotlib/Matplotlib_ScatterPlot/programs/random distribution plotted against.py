from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt
from pylab import randn


# class to perform graphical representation of data using matplotlib pie chart
class RandomDistributionXY:
    choice = 0

    def scatter_plots(self):
        print()
        print("1. Draw a scatter graph taking a random distribution in X and Y and plotted against each other.")
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
                        x = input("Enter random distribution in x:")
                        # validate random value for x
                        validate = validate_num(x)
                        if validate:
                            x = int(x)
                            y = input("Enter random distribution in y:")
                            # validate random value for y
                            validate = validate_num(y)
                            if validate:
                                y = int(y)
                                # generate array of random number
                                x_value = randn(x)
                                y_value = randn(y)
                                # generate a scatter plots
                                plt.scatter(x_value, y_value, color='r')
                                plt.xlabel('X')
                                plt.ylabel('Y')
                                plt.show()
                            else:
                                print("Enter only numbers")

                        elif choice == 2:
                            exit()
                    else:
                        print("Enter valid choice")
                else:
                    print("Enter only numbers")
            except Exception as e:
                print(e)


# instantiation
RandomDistributionXY_object = RandomDistributionXY()
RandomDistributionXY_object.scatter_plots()
