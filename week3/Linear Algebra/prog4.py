"""4. Write a program to multiply matrices in problem 1"""
from week3.Utility.utility import Utility1


class Multiplication(Utility1):

    # constructor for subclass

    def __init__(self):
        super(Multiplication, self).__init__()
        Utility1.read_matrix1(self)
        Utility1.read_matrix2(self)
        self.result_list = [[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]

    def call(self):
        print("\n Vector Multiplication:", obj1.mul_matrix(self.mat_list1, self.mat_list2, self.result_list))


obj1 = Utility1()
# object of subclass

obj = Multiplication()
# call method
obj.call()
