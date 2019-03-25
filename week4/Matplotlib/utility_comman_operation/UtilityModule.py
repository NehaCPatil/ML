from week4.Matplotlib.Matplotlib_PieChart.Utility.piechart_utility import *


class Utility:
    xrange = []

    def get_range(self):
        while 1:
            # get starting range
            range_from = input("Range from :")
            # validate range
            valid_range1 = validate_num(range_from)
            if valid_range1:
                # if valid then get ending range
                range_from = int(range_from)
                range_to = input("Range to:")
                # validate ending range
                valid_range2 = validate_num(range_to)
                if valid_range2:
                    range_to = int(range_to)
                    # store start and end range in a list
                    self.xrange = range(range_from, range_to)
                    # pass list of range to function
                    return self.xrange
                else:
                    print("Invalid range")
            else:
                print("Enter valid range")
                # getting choice of user to continue
                ans = input("You want to continue..[y/n]")
                if ans == 'n' or ans == 'N':
                    return "exit..."


