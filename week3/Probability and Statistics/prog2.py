"""2. Write a program to find the probability of drawing an ace after drawing a king on the first draw """

from week3.Utilitystatic.util import Utility1


class Probability:
    def __init__(self):
        self._king = 4
        self.cards = 52
        self.cards_drawn = 1
        self._cards = self.cards - self.cards_drawn

    # Determine the probability of drawing an Ace after drawing a King on the first draw

    def call_function(self):
        king_prob = obj.probability_cardsK(self._king, self._cards)

        print("\n probability of drawing an ace from pack of cards:", king_prob)

        print(round(king_prob, 2))

        ace_probability_percent = king_prob * 100

        print("\n ace_probability_percent: ")
        
        print(str(round(ace_probability_percent, 0)) + '%')


obj = Utility1()

obj1 = Probability()

obj1.call_function()



