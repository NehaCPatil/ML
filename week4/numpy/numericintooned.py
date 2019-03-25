
from week4.numpy.Utility.util import Utility


# class
class Numpy:

    # constructor
    def __init__(self):
        self.list = [12.23, 13.32, 100, 36.32]
        self.obj = Utility()

    # function for display array
    def convert_display(self):
        # call function
        array1 = self.obj.convert_to_one_d(self.list)
        # display array
        print("one-dimensional NumPy array", array1)

    # function for display matrix

    def mat_display(self):
        # call function
        matrix1 = self.obj.create_matrix()
        # print matrix
        print(matrix1)

    # display vector

    def vector(self):
        # call function and display vector
        print("vector", self.obj.update_vector())

    # display reverse array

    def reverse_display(self):
        # call function reverse and display reverse array
        print("reverse an array", self.obj.reverse_array())

    # display 2d array

    def array_two_display(self):
        # call function two d array
        print(self.obj.two_darray())

    # display add border array

    def add_border_display(self):
        # call function
        print(self.obj.add_border())

    # display checkerboard

    def display_checkerboard(self):
        # call function
        print(self.obj.print_check_board())

    # display convert list and tuple into array

    def display_list_tuple_array(self):
        # call function
        print(self.obj.convert_list_array())

    # display append value array
    def display_append_array(self):
        # call function append array
        self.obj.append_value_end_array()

    # display imaginary and real part
    def display_real_imaginary(self):
        # call function
        self.obj.real_imaginary_part()


# problem statement
print("_______________________________________________________________________________________________________________")
print("1. Get one-dimensional NumPy array ")
print("2. Create a 3x3 matrix with values ranging from 2 to 10.  ")
print("3. Write a Python program to create a null vector of size 10 and update sixth value to 11. ")
print("4. Write a Python program to reverse an array (first element becomes last")
print("5. Write a Python program to create a 2d array with 1 on the border and 0 inside.  ")
print("6. Write a Python program to add a border (filled with 0's) around an existing array.")
print("7. Write a Python program to create a 8x8 matrix and fill it with a checkerboard pattern.")
print("8. Write a Python program to convert a list and tuple into arrays. ")
print("9. Write a Python program to append values to the end of an array. ")
print("10. Write a Python program to find the real and imaginary parts of an array of complex numbers. ")

object1 = Numpy()
while 1:
    try:
        print("_______________________________________________________________________________________________________")
        ch = input("\n\nEnter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:

                object1.convert_display()

            elif choice == 2:
                object1.mat_display()

            elif choice == 3:
                object1.vector()

            elif choice == 4:
                object1.reverse_display()

            elif choice == 5:
                object1.array_two_display()

            elif choice == 6:
                object1.add_border_display()

            elif choice == 7:
                object1.display_checkerboard()

            elif choice == 8:
                object1.display_list_tuple_array()

            elif choice == 9:
                object1.display_append_array()

            elif choice == 10:
                object1.display_real_imaginary()

            elif choice == 11:
                exit()

        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")
