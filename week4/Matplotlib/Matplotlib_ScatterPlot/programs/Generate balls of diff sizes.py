from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt
import random
import math

# class to perform graphical representation of data using matplotlib pie chart
class GenerateBallsOfDifferentSizes:
    choice = 0

    def scatter_plots(self):
        print()
        print("1. Draw a scatter plot using random distributions to generate balls of different sizes.")
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
                        no_of_balls = 25
                        # Draw samples from the triangular distribution over the interval [left, right].
                        x = [random.triangular() for _ in range(no_of_balls)]
                        # draw samples from normal distribution/gaussian distribution
                        y = [random.gauss(0.5, 0.25) for _ in range(no_of_balls)]
                        colors = [random.randint(1, 4) for _ in range(no_of_balls)]
                        areas = [math.pi * random.randint(5, 15) ** 2 for _ in range(no_of_balls)]
                        # create a figure object
                        plt.figure()
                        plt.scatter(x, y, s=areas, c=colors, alpha=0.850)
                        plt.axis([0.0, 1.0, 0.0, 1.0])
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.show()
                    elif choice == 2:
                        exit()
                else:
                    print("Enter valid choice")
            except Exception as e:
                print(e)


# instantiation
GenerateBallsOfDifferentSizes_obj = GenerateBallsOfDifferentSizes()
GenerateBallsOfDifferentSizes_obj.scatter_plots()
