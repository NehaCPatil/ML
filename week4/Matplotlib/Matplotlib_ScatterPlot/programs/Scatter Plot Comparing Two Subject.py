from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt


# class to perform graphical representation of data using matplotlib pie chart
class ScatterPlotCompareTwoSubject:
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
                        # range of marks
                        marks_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                        print("Mathematics marks:")
                        # get maths marks
                        math_marks = get_marks(len(marks_range))
                        print("Science marks:")
                        # get science marks
                        science_marks = get_marks(len(marks_range))
                        # plot math marks
                        plt.scatter(marks_range, math_marks, label="Maths marks", color='g')
                        # plot science marks
                        plt.scatter(math_marks, science_marks, label="Science marks", color='r')
                        plt.xlabel("marks range")
                        plt.ylabel("marks scored")
                        # linked to the data being graphically displayed in the plot area of the chart.
                        plt.legend()
                        plt.show()
                    elif choice == 2:
                        exit()
                else:
                    print("Enter valid choice")
            except Exception as e:
                print(e)


# class object
ScatterPlotCompareTwoSubject_obj = ScatterPlotCompareTwoSubject()
ScatterPlotCompareTwoSubject_obj.scatter_plots()
