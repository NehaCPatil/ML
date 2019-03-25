from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt
import numpy as np


# class to perform graphical representation of data using matplotlib pie chart
class CompareWeightsAndHeights:
    choice = 0

    def scatter_plots(self):
        print()
        print("1. Draw a scatter plot comparing two subject marks of Mathematics and Science.Use marks of 10 students.")
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
                        print("Weight 1")
                        weight1 = create_list_all(10)
                        print("height 1:")
                        height1 = create_list_all(10)
                        print("Weight 2")
                        weight2 = create_list_all(10)
                        print("height 2:")
                        height2 = create_list_all(10)
                        print("Weight 3")
                        weight3 = create_list_all(10)
                        print("height 3:")
                        height3 = create_list_all(10)
                        weight = np.concatenate(weight1, weight2, weight3)
                        height = np.concatenate(height1, height2, height3)
                        plt.scatter(weight, height, marker='*', color=['red', 'green', 'blue'])
                        plt.xlabel("weight", fontsize=16)
                        plt.ylabel("height", fontsize=16)
                        plt.title("Group wise weight and height scatter plot", fontsize=20)
                        plt.show()

                    elif choice == 2:
                        exit()
                else:
                    print("Enter valid choice")
            except Exception as e:
                print(e)


# instantiation
CompareWeightsAndHeights_obj = CompareWeightsAndHeights()
CompareWeightsAndHeights_obj.scatter_plots()
