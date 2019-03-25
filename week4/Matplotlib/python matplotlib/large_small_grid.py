# 14. Write a Python program to display the grid and draw line charts of the closing value of Alphabet Inc.
# between October 3, 2016 to October 7, 2016. Customized the grid lines with rendering with a larger grid
# (major grid) and a smaller grid (minor grid).Turn on the grid but turn off ticks.
import datetime as DT
from matplotlib import pyplot as plt
from matplotlib.dates import date2num


class large_small_grid:

    def draw_line(self):
        data = [(DT.datetime.strptime('2016-10-03', "%Y-%m-%d"), 772.559998),
                (DT.datetime.strptime('2016-10-04', "%Y-%m-%d"), 776.429993),
                (DT.datetime.strptime('2016-10-05', "%Y-%m-%d"), 776.469971),
                (DT.datetime.strptime('2016-10-06', "%Y-%m-%d"), 776.859985),
                (DT.datetime.strptime('2016-10-07', "%Y-%m-%d"), 775.080017)]

        # Convert datetime objects to Matplotlib dates.
        x = [date2num(date) for (date, value) in data]
        y = [value for (date, value) in data]

        # creates object of figure class
        fig = plt.figure()

        graph = fig.add_subplot(1, 1, 1)

        # Plot the data as a red line with round markers
        graph.plot(x, y, 'r-o')

        # Set the xtick locations
        graph.set_xticks(x)

        # Set the xtick labels
        graph.set_xticklabels(
            [date.strftime("%Y-%m-%d") for (date, value) in data]
        )

        # Set the x axis label
        plt.xlabel('Date')
        # Set the y axis label
        plt.ylabel('Closing Value')
        # Sets a title
        plt.title('Closing stock value of Alphabet Inc.')

        # Turn on the minor TICKS, which are required for the minor GRID
        plt.minorticks_on()

        # Customize the major grid
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        # Customize the minor grid
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        # Turn off the display of all ticks.
        plt.tick_params(which='both',  # Options for both major and minor ticks
                        top='False',  # turn off top ticks
                        left='False',  # turn off left ticks
                        right='False',  # turn off right ticks
                        bottom='False')  # turn off bottom tfcks

        # shows the plot
        plt.show()


# creates class object
obj = large_small_grid()
# calling method by using class object
obj.draw_line()
