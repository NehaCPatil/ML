"""1. Write a Python program to create and display a one-dimensional array-like object containing an array of
data using Pandas module. """

from week4.panda.Utility.Util import Utility


class PandaPrograms:

    def __init__(self):
        self.utility_obj = Utility()

    print("1. display a one-dimensional array-like object containing an array of data using Pandas module.. ")
    print("2. convert a Panda module Series to Python list and it's type. ")
    print("3. Python program to add, subtract, multiple and divide two Pandas Series.")
    print("4. program to get the powers of an array values element-wise")
    print("5. display a DataFrame from a specified dictionary data which has the index labels")
    print("6. display a summary of the basic information about a specified Data Frame and its data ")
    print("7. get the first 3 rows of a given DataFrame. Sample Python dictionary  data and list labels")
    print("8. Write a Python program to select the 'name' and 'score' columns from the following DataFrame.")
    print("9.  Write a Python program to select the specified columns and rows from a given data frame.")
    print("10. select the rows where the number of attempts in the examination is greater than 2.")
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
                    """1. Write a Python program to create and display a one-dimensional array-like object containing an
                     array of data using Pandas module."""

                    n = int(input("How many element you want to add:"))
                    num = int(n)
                    data = self.utility_obj.series_data(num)
                    print(data)

                elif choice == 2:
                    # 2. Write a Python program to convert a Panda module Series to Python list and it's type.

                    n = int(input("How many element you want to add:"))
                    num = int(n)
                    data = self.utility_obj.series_data(num)
                    print(data)
                    self.utility_obj.convert_series_to_list(data)


                elif choice == 3:

                    # 3. Write a Python program to add, subtract, multiple and divide two Pandas Series.

                    n = int(input("How many element you want to add:"))
                    num = int(n)
                    data = self.utility_obj.series_data(num)
                    print(data)

                    n = int(input("How many element you want to add:"))
                    num = int(n)
                    data1 = self.utility_obj.series_data(num)
                    print(data)

                    
                    my_data = self.utility_obj.operation_on_series(data, data1)


                elif choice == 4:

                    # 4. Write a Python program to get the powers of an array values element-wise.

                    n = int(input("\n How many element you want to add:"))
                    num = int(n)
                    data = self.utility_obj.series_data(num)
                    print(data)
                    print("\n First array elements raised to powers from second array, element-wise:")
                    my_data = self.utility_obj.powers_of_an_array(data)

                elif choice == 5:

                    """5. Write a Python program to create and display a DataFrame from a specified dictionary data which
                     has the index labels."""

                    my_data_frame = self.utility_obj.dataFrame_from_dictionary()


                elif choice == 6:
                    """6. Write a Python program to display a summary of the basic information about a specified Data Frame 
                    and its data."""

                    data_frame = self.utility_obj.basic_Information_About_Data_Frame()

                elif choice == 7:
                    """7. Write a Python program to get the first 3 rows of a given DataFrame. Sample Python dictionary 
                    data and list labels:"""

                    data_frame = self.utility_obj.first_3_rows_of_DataFrame()


                elif choice == 8:
                    """8. Write a Python program to select the 'name' ,'score' columns from the following DataFrame"""

                    data_frame = self.utility_obj.select_name_and_score_columns()

                elif choice == 9:
                    """9. Write a Python program to select the specified columns and rows from a given data frame. """

                    data_frame = self.utility_obj.specified_Columns_and_rows()

                elif choice == 10:
                    """10. Write a Python program to select the rows where the number of attempts in the examination is 
                    greater than 2. """

                    data_frame = self.utility_obj.select_rows_with_attempt_more_than_2()

                else:

                    print("\n Enter Valid choice between 0- 10")

            except Exception as e:
                print("Invalid input")


panda = PandaPrograms()
panda.while_display()
