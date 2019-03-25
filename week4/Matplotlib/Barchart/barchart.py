
"""11. Write a Python program to create bar plot from a DataFrame.
Sample Data Frame:
a b c d e
2 4,8,5,7,6
4 2,3,4,2,6
6 4,7,4,7,8
8 2,6,4,8,6
10 2,4,3,3,2

12. Write a Python program to create bar plots with error bars on the same figure. Sample Date
Mean velocity: 0.2474, 0.1235, 0.1737, 0.1824
Standard deviation of velocity: 0.3314, 0.2278, 0.2836, 0.2645


"""
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


class BarGraph:

    def createarr(self,size):
        try:
            empty_list = list()

            # we have to typecast num to compare with length of string
            num2 = int(size)
            # checking enter value is only digit or not
            if size.isdigit():
                print("Enter the elements: ")
                for ele in range(num2):
                    res = int(input())
                    empty_list.append(res)
                    # in array i is -> signed integer, f-> float(size 4 byte), d ->float(size 8)

                print("list Elements:", empty_list)
                return empty_list
            else:
                raise ValueError
        except ValueError:
            print("Enter valid number: ")

    def barchart(self):
            size = input("Enter the size(it must be 5) for list:")

            print("Enter 1st list elements")
            l1 = self.createarr(size)
            # print(l1)
            print("Enter 2st list elements")
            l2 = self.createarr(size)
            # print(l2)
            print("Enter 3rd list elements")
            l3 = self.createarr(size)
            print("Enter 4th list elements")
            l4 = self.createarr(size)
            print("Enter 5th list elements")
            l5 = self.createarr(size)

            # convert list to array
            data = np.array([l1, l2, l3, l4, l5])

            # arrange all data in a Data Frame, here
            df = DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'], index=[2, 4, 6, 8, 10])
            # kind check , which type of graph, kind='bar'-> specify vertical bar (by default)
            df.plot(kind='bar')

            plt.show()

    def error_bar(self):
        N = 5
        menMeans = (54.74, 42.35, 67.37, 58.24, 30.25)
        menStd = (4, 3, 4, 1, 5)

        # the x locations for the groups
        ind = np.arange(N)

        # the width of the bars
        width = 0.35

        # x(as ind), y(menMeans) define the data locations, yerr define the errorbar sizes.
        plt.bar(ind, menMeans, width, yerr=menStd, color='blue')

        # fig, ax = plt.subplots()
        # rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

        # put x and y axis label
        plt.ylabel('Scores')
        plt.xlabel('Velocity')

        # set graph title
        plt.title('Scores by Velocity, Error Bar')

        plt.show()

# create class object
obj = BarGraph()
obj.barchart()
obj.error_bar()
