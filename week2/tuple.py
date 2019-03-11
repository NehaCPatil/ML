
from week2.Utility.Util import Utility

class Tuple:

    obj = Utility()

    number = (12, 213, 2313, 55, 44, 76, 23, 34, 789)

    flag = True

    print("1. Python program to create a tuple. ")
    print("2. Python program to create a tuple with different data types ")
    print("3. Python program to unpack a tuple in several variables.")
    print("4. Python program to create the colon of a tuple.")
    print("5. Python program to find the repeated items of a tuple")
    print("6. Python program to check whether an element exists within a tuple. ")
    print("7. Python program to convert a list to a tuple. ")
    print("8. Python program to remove an item from a tuple.")
    print("9. Python program to slice a tuple.")
    print("10.Python program to reverse a tuple")
    print("0. EXIT")

    while flag:

        try:
            print('____________________________________________________________________________________')

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            elif choice == 1:

                """1. Python program to create a tuple."""

                print("\n Tuple:", obj.tuple_create())

            elif choice == 2:

                """2. Python program to create a tuple with different data types"""

                print("\n tuple with different data types", obj.tuple_diff())

            elif choice == 3:

                """3. Python program to unpack a tuple in several variables."""

                tuple1 = ("IIM", 5000, "Engineering")

                # this lines UNPACKS values
                (college, student, type_of_college) = tuple1

                # print
                print(college)
                print(student)
                print(type_of_college)

            elif choice == 4:

                """4. Python program to create the colon of a tuple."""

                li1 = [4, 8, 2, 10, 15, 18]

                print("\n\n Original List:", li1)
                print("\n\n After Cloning:", obj.Cloning(li1))

            elif choice == 5:

                """5. Python program to find the repeated items of a tuple"""

                vowels = ('a', 'e', 'i', 'o', 'i', 'o', 'e', 'i', 'u')

                print("\n\n Repeated element:", obj.repet_element(vowels))

            elif choice == 6:

                """6. Python program to check whether an element exists within a tuple."""

                number = (12, 213, 2313, 55, 44, 76, 23, 34, 789)

                try:
                    pos = number.index(55)
                    print("\n\n Element a Found at position : ", pos)

                except ValueError as e:
                    print(e)

            elif choice == 7:

                """7. Python program to convert a list to a tuple. """

                list = [1, 34, 435, 546]

                print("\n convert a list to a tuple", tuple(list))

            elif choice == 8:

                """8. Python program to remove an item from a tuple."""

                ind = 3

                # number.pop(55)
                try:

                    number = number[: ind] + number[ind + 1:]

                except IndexError:

                    print("Enter index not found")

                print("\n\n Modified Tuple : ", number)

            elif choice == 9:

                """9. Python program to slice a tuple."""

                print("\n\n slice of tuple ", number[2:])

            elif choice == 10:

                """10.Python program to reverse a tuple"""

                print("\n\nOriginal Tuple", number)

                print("Reverse Tuple", obj.Reverse_tuple(number))

            else:
                print("enter valid choice between 0-10")

        except Exception as e:
            print(e)
