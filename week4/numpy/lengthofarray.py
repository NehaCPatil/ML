from week4.numpy.Utility.util import Utility


class Array:

    # constructor

    def __init__(self):
        self.object1 = Utility()

    # display length array

    def display_lenght_array(self):
        # call method
        self.object1.length_of_array()

    # display common element

    def display_common(self):
        # call method
        self.object1.common_value_array()

    # display set diff

    def display_set_diff(self):
        # call method
        print(self.object1.set_diff_array())

    # display set exclusive

    def display_set_exclusive(self):
        # call method
        print("Set Exclusive:", self.object1.set_exclusive())

    # display compare two array

    def display_compare_array(self):
        # call method
        self.object1.compare_array()

    # display txt file

    def display_txt_arr(self):
        # call method
        self.object1.save_array_txt()

    # display flattened array

    def display_flattened(self):
        # call
        self.object1.contiguous_flattened_array()

    def display_change_dtype_array(self):
        print(self.object1.change_type_array())

    def display_diagonal_mat(self):
        self.object1.create_diagonal_mat()

    def display_upper_triangular(self):
        print(self.object1.upper_triangular())


# problem statement
print("_______________________________________________________________________________________________________________")
print("11.number of elements of an array,length of one array element in bytes and total bytes consumed by the elements")
print("12. Write a Python program to find common values between two arrays. ")
print("13.set difference of two arrays.set difference return the sorted,unique values in arr1 that are not in arr2. ")
print("14.Set exclusive-or will return the sorted, unique values that are in only one (not both) of the input arrays. ")
print("15. Write a Python program compare two arrays using numpy. ")
print("16. Write a Python program to save a NumPy array to a text file. ")
print("17. Write a Python program to create a contiguous flattened array. ")
print("18. Write a Python program to change the data type of an array. ")
print("19. Write a Python program to create a 3-D array with ones on a diagonal and zeros elsewhere. ")
print("20.  Write a Python program to create an array which looks like below array. ")

obj = Array()
while 1:
    try:
        # print("_______________________________________________________________________________________________________")
        ch = input("\n\nEnter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:
                obj.display_lenght_array()

            elif choice == 2:
                obj.display_common()

            elif choice == 3:
                obj.display_set_diff()

            elif choice == 4:
                obj.display_set_exclusive()

            elif choice == 5:
                obj.display_compare_array()

            elif choice == 6:
                obj.display_txt_arr()

            elif choice == 7:
                obj.display_flattened()

            elif choice == 8:
                obj.display_change_dtype_array()

            elif choice == 9:
                obj.display_diagonal_mat()

            elif choice == 10:
                obj.display_upper_triangular()

            elif choice == 11:
                exit()

        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")
