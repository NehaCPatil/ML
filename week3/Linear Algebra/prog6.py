"""	6. Write a program to find transpose matrix of matrix Y in problem 1"""

from week3.Utility.utility import Utility1


class Transpose(Utility1):

    # constructor for subclass

    def __init__(self):
        super(Transpose, self).__init__()
        Utility1.read_matrix1(self)

    # function for transpose
    def call(self):
        print(obj1.transpose_mat(self.mat_list1))


obj1 = Utility1()

# object of subclass
obj = Transpose()

# call method
obj.call()
