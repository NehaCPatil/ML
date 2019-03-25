import numpy as np
from array import *


class Utility:

    # convert a list of numeric value into a one-dimensional NumPy array

    def convert_to_one_d(self, list1):
        my_array = np.asarray(list1)
        return my_array

    # 3x3 matrix

    def create_matrix(self):
        matrix1 = np.arange(2, 11)
        mat = matrix1.reshape(3, 3)
        return mat

    # null vector of size 10 and update sixth value to 11

    def update_vector(self):
        matrix1 = np.zeros(10)
        for _ in matrix1:
            matrix1[6] = 11
        return matrix1

    # reverse an array

    def reverse_array(self):
        array1 = np.arange(12, 37)
        print(array1)
        return array1[::-1]

    # create a 2d array with 1 on the border and 0 inside.

    def two_darray(self):
        # create array 5 * 5
        array1 = np.ones((5, 5))

        # display create array

        print("\n Original array:\n ")
        print(array1)
        print("\n 1 on the border and 0 inside in the array \n")

        # 1 on the border and 0 inside in the array
        array1[1:-1, 1:-1] = 0
        return array1

    # add a border (filled with 0's)

    def add_border(self):
        # create array
        my_array = np.ones((3, 3))
        print("\n Original array:\n")
        print(my_array)
        print("\n 0 on the border and 1 inside in the array\n")
        # add a border (filled with 0's)
        my_array = np.pad(my_array, pad_width=1, mode='constant', constant_values=0)
        return my_array

    # function to print Checkerboard pattern

    def print_check_board(self):
        print("Checkerboard pattern:")
        # create a n * n matrix
        x = np.zeros((8, 8), dtype=int)
        # fill with 1 the alternate rows and columns
        x[1::2, ::2] = 1
        x[::2, 1::2] = 1
        return x

    def convert_list_array(self):
        # create list and tuple

        my_list = [1, 3, 5, 7, 9]
        my_tuple = ([1, 3, 9], [8, 2, 6])

        # print create list  and tuple

        print("\nInput list : ", my_list)
        print(" \nInput tuple: ", my_tuple)

        # convert list into array

        out_arr = np.asarray(my_list)

        # convert tuple in array

        out_tuple = np.asarray(my_tuple)
        return out_arr, out_tuple

    def append_value_end_array(self):
        array_num = array('i', [1, 3, 5, 7, 9])
        print("Original array: " + str(array_num))
        print("Append 11 at the end of the array:")
        array_num.append(11)
        print("New array: " + str(array_num))

    def real_imaginary_part(self):
        x = np.sqrt([1 + 0j])
        y = np.sqrt([0 + 1j])
        print("Original array:x ", x)
        print("Original array:y ", y)
        print("Real part of the array:")
        print(x.real)
        print(y.real)
        print("Imaginary part of the array:")
        print(x.imag)
        print(y.imag)

    def length_of_array(self):
        x = np.array([1, 2, 3], dtype=np.float64)
        print("Size of the array: ", x.size)
        print("Length of one array element in bytes: ", x.itemsize)
        print("Total bytes consumed by the elements of the array: ", x.nbytes)

    def common_value_array(self):
        print(np.intersect1d([0, 10, 20, 40, 60], [10, 30, 40]))

    # set difference of two array
    def set_diff_array(self):
        a = np.array([0, 10, 20, 40, 60, 80])
        b = np.array([10, 30, 40, 50, 70, 90])
        return np.setdiff1d(a, b)

    # set Set exclusive

    def set_exclusive(self):
        array1 = np.array([0, 10, 20, 40, 60, 80])
        print("Array1: ", array1)
        array2 = np.array([10, 30, 40, 50, 70])
        print("Array2: ", array2)
        print("Unique values that are in only one (not both) of the input arrays:")
        return np.setxor1d(array1, array2)

    # compare two array
    def compare_array(self):
        a = np.array([1, 2])
        b = np.array([4, 5])
        print("Array a: ", a)
        print("Array b: ", b)
        print("a > b")
        print(np.greater(a, b))
        print("a >= b")
        print(np.greater_equal(a, b))
        print("a < b")
        print(np.less(a, b))
        print("a <= b")
        print(np.less_equal(a, b))

    # text file
    def save_array_txt(self):
        a = np.array([1, 2])
        fo = open("myarray.txt", "r+")
        print("Name of the file: ", fo.name)

        line = fo.read(10)
        print("Read Line: %s" % (line))

        return np.savetxt("myarray.txt", a, delimiter=" ")

    # create a contiguous flattened array.

    def contiguous_flattened_array(self):
        a = np.array([[1, 2], [3, 4]])
        print(a.flatten())

    def change_type_array(self):
        arr = np.array([[2, 4, 6], [6, 8, 10]], dtype=int)
        return arr.astype(np.float64)

    # function for Diagonal matrix

    def create_diagonal_mat(self):
        # define array
        arr = np.array([7, 7, 7])
        # for
        mat = np.diag(arr)
        print(mat)

    # function for Upper triangular

    def upper_triangular(self):
        return np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], -1)

    # concatenate two 2-dimensional arrays.

    def concatenate_array(self):
        # define array
        a = np.array([[0, 1, 3], [5, 7, 9]])
        b = np.array([[0, 2, 4], [6, 8, 10]])
        # concatenate two array
        c = np.concatenate((a, b), 1)
        return c

    # make an array immutable

    def array_immutable(self):
        # define array
        x = np.zeros(10)
        # set flag to writeable false for read only
        x.flags.writeable = False
        print("Test the array is read-only or not:")
        print("Try to change the value of the first element:")
        x[0] = 1

    #
    def multiply_every_element(self):
        a = np.array([1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1 ])
        mat = a.reshape(3, 4)
        print(mat * 3)

    # convert numpy to list
    def array_to_list(self):
        return np.array([[1, 2, 3], [4, 5, 6]]).tolist()


    def scientific_notation(self):
        arr = np.array([1.60000000e-10, 1.60000000e+00, 1.20000000e+03, 2.35000000e-01])
        print("Original array elements:")
        print(arr)
        print("Print array values with precision 3:")
        np.set_printoptions(suppress=True)
        print(arr)

    def extra_column(self):
        x = np.array([[10, 20, 30], [40, 50, 60]])
        y = np.array([[100], [200]])
        print(np.append(x, y, axis=1))

    def delete_element(self):
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        index = [0, 3, 4]
        print("Original array:")
        print(x)
        print("Delete first, fourth and fifth elements:")
        new_x = np.delete(x, index)
        return new_x








