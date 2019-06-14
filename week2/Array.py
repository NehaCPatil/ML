from week2.Utility.Util import Utility
from array import *

class Arr:

    # create utility object
    obj = Utility()

    arr = array('i', [10, 20, 30, 40, 50])

    flag = True

    print("1. Access individual element through indexes. ")
    print("2. program to reverse the order of the items in the arra. ")
    print("3. program to get the number of occurrences of a specified element in an array.")
    print("4.  program to remove the first occurrence of a specified element from an array.")
    print("0. EXIT")

    while flag:

        try:

            print('_________________________________________________________________________________________________')

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            elif choice == 1:
                """1. Python program to create an array of 5 integers and display the array items 
                Access individual element through indexes. """

                # return all element of array
                print("\n\n array :", obj.display_arr(arr))

                try:
                    # return element of array on specific index
                    print("\n 1st element of Array", arr[0])

                    print("\n 2nd element of Array", arr[1])

                    print("\n 3rd element of Array", arr[2])

                    print("\n 4th element of Array", arr[3])

                    print("\n 5th element of Array", arr[4])

                    print("\n 6th element of array", arr[5])

                except IndexError:
                    print("Index out of bound error.")

                else:
                    print("Success")

            elif choice == 2:

                """ 2. Python program to reverse the order of the items in the array."""

                print("\n\n array :", arr)

                print("\n reverse the order of the items in the array", obj.arr_revers(arr))

            elif choice == 3:

                """ 3. program to get the number of occurrences of a specified element in an array"""

                print("\n\n array :", arr)
                try:

                    print("\n occurrences of a specified element in an array", obj.arr_count(arr))

                except ValueError:
                    print("\n An error occurred")

            elif choice == 4:

                """ 4. Python program to remove the first occurrence of a specified element from an array"""

                print("\n\n array :", arr)

                try:
                    print("\n remove the first occurrence of a specified element", obj.arr_remove(arr))

                except ValueError:
                    print("\n element is not found ")

            else:
                print("\n Enter Valid choice between 0-4")

        except Exception as e:
            print(e)
