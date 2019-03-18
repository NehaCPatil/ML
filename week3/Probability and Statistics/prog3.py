""" 3.Write a program to find the probability of drawing an ace after drawing an ace on the first draw """

from week3.Utilitystatic.util import Utility1


class Probability:

    def call_function(self):
        drawnA = obj.drawn_aces()
        drawnCards = obj.cards()
        probability_cardsA = drawnA / drawnCards
        print(probability_cardsA)


obj = Utility1()
obj1 = Probability()
print("probability of drawing an ace after drawing an ace")
obj1.call_function()



