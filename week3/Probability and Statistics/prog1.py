"""1. Write a program to find probability of drawing an ace from pack of cards """

from week3.Utilitystatic.util import Utility1


class Probability:
    def __init__(self):
        self._aces = 4
        self._cards = 52

    def call_function(self):
        aces_prob = obj.probability_cards(self._aces, self._cards)

        print("\n probability of drawing an ace from pack of cards:", aces_prob)

        print(round(aces_prob, 2))

        ace_probability_percent = aces_prob * 100

        print("\n ace_probability_percent: ")
        print(str(round(ace_probability_percent, 0)) + '%')



obj = Utility1()

obj1 = Probability()

obj1.call_function()
