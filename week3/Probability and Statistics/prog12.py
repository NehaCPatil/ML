"""12.  Write a program to find the probability of getting a random number from the interval [2, 7]"""

import random


class Random:
    def __init__(self):
        self.rangediff = 6
        self.possibility=1

    def generateNum(self):
        num = random.randint(2, 7)
        print("random number is ", num)
        obj.display(num)

    def display(self, num):
        print("probability of getting a random number", num, " =", self.possibility/self.rangediff)


obj = Random()
obj.generateNum()

