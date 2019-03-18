"""1. Write a python program to add below matrices """
from week3.Utility.utility import Utility1


class Addition(Utility1):

    # constructor of subclass
    def __init__(self):
        super(Addition, self).__init__()
        Utility1.read_matrix1(self)
        Utility1.read_matrix2(self)

    def call(self):
        print("\n Addition Matrix:-", obj1.add_matrix(self.mat_list1, self.mat_list2))


# Utility object
obj1 = Utility1()

# object of subclass
obj = Addition()

# call method

obj.call()

