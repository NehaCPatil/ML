""" 5. In my town, it's rainy one third of the days. Given that it is rainy, there will be heavy traffic with
probability 12,and given that it is not rainy, there will be heavy traffic with probability 14. If it's rainy and there
 is heavy traffic,I arrive late for work with probability 12. On the other hand, the probability of being late is
 reduced to 18 if it is not rainy and there is no heavy traffic. In other situations (rainy and no traffic, not rainy
 and traffic) the probability of being late is 0.25. You pick a random day.
Write a program to find following
1. What is the probability that it's not raining and there is heavy traffic and I am not late?
2. What is the probability that I am late?
3. Given that I arrived late at work, what is the probability that it rained that day? """

from week3.Utilitystatic.util import Utility1


class Rainy:

    def __init__(self):
        self.rainy = 1 / 3
        self.rainy_with_Traffic = 1 / 2
        self.rainy_withTraffic_late = 1 / 2
        self.rainy_withTraffic_Nolate = 1 / 2

        self.rainy_with_NoTraffic = 1 / 2
        self.rainy_with_NoTraffic_late = 1 / 4
        self.rainy_with_NoTraffic_Nolate = 3 / 4

        self.not_rainy = 2 / 3
        self.not_rainy_with_Traffic = 1 / 4
        self.not_rainy_with_Traffic_late = 1 / 4
        self.not_rainy_with_Traffic_Nolate = 3 / 4

        self.not_rainy_with_NoTraffic = 3 / 4
        self.not_rainy_with_NoTraffic_late = 1 / 8
        self.not_rainy_with_NoTraffic_Nolate = 7 / 8

    # 1. What is the probability that it's not raining and there is heavy traffic and I am not late?

    def display(self):
        prob = obj1.prob_notRainy_traffic_notlate(self.not_rainy, self.not_rainy_with_Traffic,
                                                  self.not_rainy_with_Traffic_Nolate)
        print("\n probability that it's not raining and there is heavy traffic and I am not late :", prob)

    # 2. What is the probability that I am late?

    def late(self):
        rainy_traffic_late = self.rainy * self.rainy_with_Traffic * self.rainy_withTraffic_late
        rainy_notraffic_late = self.rainy * self.rainy_with_NoTraffic * self.rainy_with_NoTraffic_late
        norainy_traffic_late = self.not_rainy * self.not_rainy_with_Traffic * self.not_rainy_with_Traffic_late
        norainy_notraffic_late = self.not_rainy * self.not_rainy_with_NoTraffic * self.not_rainy_with_NoTraffic_late

        return rainy_traffic_late + rainy_notraffic_late + norainy_traffic_late + norainy_notraffic_late

    # 3. Given that I arrived late at work, what is the probability that it rained that day?

    def rain_that_day(self):
        rainy_traffic_late = self.rainy * self.rainy_with_Traffic * self.rainy_withTraffic_late
        rainy_notraffic_late = self.rainy * self.rainy_with_NoTraffic * self.rainy_with_NoTraffic_late
        prob_late = object1.late()
        return rainy_traffic_late + rainy_notraffic_late / prob_late


obj1 = Utility1()
object1 = Rainy()


flag = True

print("1. What is the probability that it's not raining and there is heavy traffic and I am not late? ")
print("2. What is the probability that I am late?")
print("3. Given that I arrived late at work, what is the probability that it rained that day?")

while flag:
    try:

        print('___________________________________________________________________________________________________')

        choice = int(input("Enter your choice"))

        if choice == 0:
            flag = False

        elif choice == 1:
            print("1. What is the probability that it's not raining and there is heavy traffic and I am not late? ")
            print("\n Output of first prog :")
            object1.display()

        elif choice == 2:
            print("2. What is the probability that I am late?")
            print("\n Output of Second prog :")
            round_prob = object1.late()
            print("\n probability that I am late :", round_prob)

        elif choice == 3:
            print("3. Given that I arrived late at work, what is the probability that it rained that day?")
            print("\n Output of Third prog :")
            print("\n probability that it rained that day:", object1.rain_that_day())

        else:
            print("\n Enter Valid choice between 0-3")

    except Exception as e:
        print("Invalid input")


