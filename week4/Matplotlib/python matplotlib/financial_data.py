"""4 Write a Python program to draw line charts of the financial data of Alphabet Inc.
between October 3, 2016 to October 7, 2016."""

import matplotlib.pyplot as plt
import pandas as pd


# create class

class Financial:

    def read_data(self):

        # reading csv file
        df = pd.read_csv('test.csv', sep=',', parse_dates=True, index_col=0)

        # plotting lines
        df.plot()

        # Show the figure.
        plt.show()


# creates class object
obj = Financial()
flag = False

# calling method by using class object
obj.read_data()
