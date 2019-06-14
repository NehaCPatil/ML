
import matplotlib.pyplot as plt

# class for chart
class Diff_chart:

    def type_chart(self):
        x = range(10)
        y = range(10)

        plt.subplot(2, 2, 1)
        plt.plot(x, y)

        plt.subplot(2, 2, 2)
        plt.bar(x, y)

        plt.subplot(2, 2, 3)
        plt.hist(x, y)

        plt.subplot(2, 2, 4)

        plt.scatter(x, y)

        plt.show()


object_class = Diff_chart()
object_class.type_chart()
