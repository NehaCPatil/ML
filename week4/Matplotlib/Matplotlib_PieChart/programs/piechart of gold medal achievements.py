from week4.Matplotlib.utility_comman_operation.UtilityModule import *
import matplotlib.pyplot as plt
import pandas as pd


# class to perform graphical representation of data using mat plot lib

class ChartGoldMedal:
    choice = 0

    def gold_medal_achievement(self):

        print()
        print("1. pie chart of gold medal achievements of five most successful countries in 2016 Summer Olympics.")
        print("2. Exit")
        print()

        while True:

            try:

                print()

                # accept choice from user
                self.choice = input("Enter choice : ")

                # validate choice number
                valid_choice = validate_num(self.choice)

                if valid_choice:

                    choice = int(self.choice)

                    if choice == 1:

                        # read csv file medal.csv
                        df = pd.read_csv('medal.csv')

                        # get label and medal for each respective countries
                        country = df["country"]
                        medal = df["gold_medal"]

                        # explode first country having large medal in olympics
                        explode = (0.1, 0, 0, 0, 0)
                        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

                        # plotting of csv file data
                        plt.pie(medal, explode=explode, labels=country,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)
                        plt.title("Gold medal achievements of five most successful\n"+"countries in 2016 Summer Olymp")

                        # show figure
                        plt.show()

                    elif choice == 2:
                        exit()

                    else:
                        print("Enter valid choice")

                else:
                    print("Enter only numbers")

            except Exception as e:
                print(e)


# object of class

ChartGoldMedal_object = ChartGoldMedal()

ChartGoldMedal_object.gold_medal_achievement()
