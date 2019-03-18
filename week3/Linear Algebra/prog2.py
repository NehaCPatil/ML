""" 2. Write a program to perform scalar multiplication of matrix and a number"""
from week3.Utility.utility import Utility1


class Matrix:
    obj = Utility1()

    # constructor

    def __init__(self):
        self.mat_list1 = [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]
        self.scalar = 4

    # function for display matrix

    def read_matrix(self):
        print("mat_list:")
        for temp1 in self.mat_list1:
            print(temp1)


# object of class
obj = Matrix()

# call method
obj.read_matrix()


class ScalarMultiplication(Matrix):

    # constructor for subclass

    def __init__(self):
        super(ScalarMultiplication, self).__init__()

    def call(self):
        scalar_list = self.obj.scalar_Product_Mat(self.mat_list1, self.scalar)
        print("\n Scalar Product Matrix is : ")
        for temp1 in scalar_list:
            print(temp1)


obj = ScalarMultiplication()
obj.call()
