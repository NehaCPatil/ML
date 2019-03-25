from week4.panda.Utility.Util import Utility


class PandaPrograms:

    def __init__(self):
        self.utility_obj = Utility()

        print("1. Write a Python program to count the number of rows and columns of a DataFrame.  ")
        print("2. Write a Python program to select the rows where the score is missing, ")
        print("3. select the rows where number of attempts in the examination is less than 2 and score greater than 15")
        print("4. Write a Python program to change the score in row 'd' to 11.5. ")
        print("5. Write a Python program to calculate the sum of the examination attempts by the students. ")
        print("6. Write a Python program to calculate the mean score for each different student in DataFrame.  ")
        print("7. append a new row 'k' to data frame with each column.delete the new row and return the original Dataf")
        print("8. to sort the DataFrame first by 'name' in descending order, then by 'score' in ascending order")
        print("9. to replace the 'qualify' column contains the values 'yes' and 'no' with True and False. .")
        print("10. Write a Python program to delete the 'attempts' column from the DataFrame. .")
        print("11. Write a Python program to insert a new column in existing DataFrame")
        print("12. Write a Python program to iterate over rows in a DataFrame.")
        print("13. Write a Python program to get list from DataFrame column headers. ")
        print("0. EXIT")

    def while_display(self):

        flag = True

        while flag:

            try:

                print()

                choice = int(input("Enter your choice"))

                if choice == 0:
                    flag = False

                elif choice == 1:
                    """11. Write a Python program to count the number of rows and columns of a DataFrame. """
                    self.utility_obj.number_of_rows_and_columns()

                elif choice == 2:
                    """12. Write a Python program to select the rows where the score is missing,"""

                    self.utility_obj.select_rows_where_score_is_missing()

                elif choice == 3:
                    """13. Write a Python program to select the rows where number of attempts in the examination is 
                    less than 2 and score greater than 15."""

                    self.utility_obj.number_attempts_less_than_2_and_score_greater_than_15()

                elif choice == 4:
                    """14. Write a Python program to change the score in row 'd' to 11.5. """
                    self.utility_obj.change_score_in_row_d()

                elif choice == 5:
                    """15. Write a Python program to calculate the sum of the examination attempts by the students. """

                    self.utility_obj.sum_of_attempts()

                elif choice == 6:
                    """16. Write a Python program to calculate the mean score for each different student in DataFrame"""

                    self.utility_obj.mean_score()

                elif choice == 7:
                    """17. Write a Python program to append a new row 'k' to data frame with given values for each 
                    column. Now delete the new row and return the original DataFrame."""

                    self.utility_obj.append_row_k_()

                elif choice == 8:
                    """18. Write a Python program to sort the DataFrame first by 'name' in descending order, then by 
                    score' in ascending order."""

                    self.utility_obj.sort_DataFrame_by_name_in_descending_and_ascending_order()

                elif choice == 9:

                    """19. Write a Python program to replace the 'qualify' column contains the values 'yes' and 'no' 
                    with True and False. """

                    self.utility_obj.replace_qualify_column_values()

                elif choice == 10:

                    """20. Write a Python program to delete the 'attempts' column from the DataFrame. """

                    self.utility_obj.delete_attempts_column()

                elif choice == 11:
                    """21. Write a Python program to insert a new column in existing DataFrame. """

                    self.utility_obj.insert_new_Column_in_existing_DataFrame()

                elif choice == 12:
                    """22. Write a Python program to iterate over rows in a DataFrame."""

                    self.utility_obj.iterate_over_rows_in_DataFrame()

                elif choice == 13:

                    """23. Write a Python program to get list from DataFrame column headers """

                    self.utility_obj.list_from_DataFrame_column_headers()

                else:

                    print("\n Enter Valid choice between 0- 13")

            except Exception:
                print("Invalid input")


panda = PandaPrograms()
panda.while_display()
