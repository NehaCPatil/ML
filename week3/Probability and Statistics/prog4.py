"""You toss a fair coin three times write a program to find following:
    1.What is the probability of three heads, HHH?
    2.What is the probability that you observe exactly one heads?
    3.Given that you have observed at least one heads, what is the probability that you observe at least two heads?"""

from week3.Utilitystatic.util import Utility1


class Probability:
    def __init__(self):
        self.sample = ['HHH', 'HHT', 'HTH', 'THH', 'HTT', 'THT', 'TTH', 'TTT']
        print("\n sample:", self.sample)
        self.len_sample = len(self.sample)
        print("\n length of sample: ", self.len_sample)

    def display(self):
        count_HHH = self.sample.count('HHH')
        print("\n count of HHH :", count_HHH)
        my_prob = obj1.count_probHHH(self.len_sample, count_HHH)
        print("\n probability of three heads : ", my_prob)
        print("_______________________________________________________________________________________________________")


obj = Probability()
obj1 = Utility1()
obj.display()

#######################################################################################################################
"""2.What is the probability that you observe exactly one heads?"""


class Exactly(Probability):

    def __init__(self):
        super(Exactly, self).__init__()

    def length_oneH(self):

        list1 = []
        for temp in self.sample:
            count = 0
            for char in temp:
                if char == 'H':
                    count += 1
            if count == 1:
                list1.append(temp)
        len_oneH = len(list1)

        print("\n one Head ", len_oneH)
        print("\n probability of exactly one heads", obj1.prob_oneH(self.len_sample, len_oneH))
        print("_______________________________________________________________________________________________________")


object1 = Exactly()
object1.length_oneH()

######################################################################################################################
"""3.Given that you have observed at least one heads, what is the probability that you observe at least two heads?"""


class Least(Probability):

    def __init__(self):
        super(Least, self).__init__()

    def display(self):
        at_least_one = obj1.at_least_one(self.sample, self.len_sample)
        at_least_two = obj1.at_least_two(self.sample, self.len_sample)
        final_prob = at_least_two / at_least_one
        print("\n probability at least two heads", final_prob)
        print("_______________________________________________________________________________________________________")


object2 = Least()
object2.display()
