# 11. Write a Python program to plot several lines with different format styles in one command using arrays.

import numpy as np
import matplotlib.pyplot as plt


class multi_format_style_lines:

    def draw_line_with_diff_formats(self):
        # Sampled time at 200ms intervals
        t = np.arange(0., 5., 0.2)

        # green dashes, blue squares and red triangles
        plt.plot(t, t, 'g--', t, t**2, 'bs', t, t**3, 'r^')
        plt.show()


# creates class object
obj = multi_format_style_lines()
# calling method by using class object
obj.draw_line_with_diff_formats()