"""6. Given the following statistics, write a program to find the probability that a woman has cancer if she has a
positive mammogram result?
a. One percent of women over 50 have breast cancer.
b. Ninety percent of women who have breast cancer test positive on mammograms.
c. Eight percent of women will have false positives."""

from week3.Utilitystatic.util import Utility1


class Cancer:
    def __init__(self):
        self.breast_cancer = 0.01
        self.No_breast_cancer = 0.99
        self.true_positive = 0.9
        self.false_positive = 0.08

    def display(self):
        object1.probability_cancer(self.true_positive, self.breast_cancer, self.false_positive, self.No_breast_cancer)


object1 = Utility1()

print("\n 1.write a program to find the probability that a woman has cancer if she has a positive mammogram result?")
while 1:
    try:
        print("\n1. Get probability \n""2. Exit")
        ch = input("Enter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:
                obj1 = Cancer()
                print("\n probability that a woman has cancer:")
                obj1.display()

            elif choice == 2:
                exit()
        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")
