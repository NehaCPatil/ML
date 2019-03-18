"""3. Write a program to perform multiplication of given matrix and vector"""

from week3.Utility.utility import Utility1


class Matrix:

    # constructor
    def __init__(self):
        self.matrix_X = [[5, 1, 3],
                         [1, 1, 1],
                         [1, 2, 1]]

        self.vector_Y = [[1], [2], [3]]

    # function for display matrix

    def read_matrix(self):
        print("mat_list:")
        for temp in self.matrix_X:
            print(temp)
        print("\nmat_list1:")
        for temp1 in self.vector_Y:
            print(temp1)

    def call(self):
        print("\n Vector Multiplication:", obj1.vector_multiply_mat(self.matrix_X, self.vector_Y))


obj1 = Utility1()

# object of class
obj = Matrix()

# call method read and call
obj.read_matrix()
obj.call()
