"""Write a Python program to create a set. """

from week2.Utility.Util import Utility


class Set:

    obj = Utility()

    set1 = {45, 66, 88, 34}

    flag = True

    print("1. Write a Python program to create a set. ")
    print("2. Write a Python program to iteration over sets. ")
    print("3. Write a Python program to add member(s) in a set. ")
    print("4. Write a Python program to remove item(s) from set ")
    print("5. Write a Python program to remove an item from a set if it is present in the set.")
    print("6. Write a Python program to create an intersection of sets.")
    print("7. Write a Python program to create a union of sets.")
    print("8. Write a Python program to create set difference.")
    print("9. Write a Python program to create a symmetric difference.")
    print("10. Write a Python program to clear a set.")
    print("11. Write a Python program to use of frozen sets.")
    print("12. Write a Python program to find maximum and the minimum value in a set. ")
    print("0. EXIT")

    while flag:

        try:

            print('_____________________________________________________________________________________')

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            elif choice == 1:

                """1. Write a Python program to create a set."""
                print("\n Set:", set1)

            elif choice == 2:

                """2. Write a Python program to iteration over sets. """
                set2 = obj.display_set(set1)

            elif choice == 3:

                """ 3. Write a Python program to add member(s) in a set"""

                print("\n Updated set:", obj.add_set(set1))

            elif choice == 4:

                """ 4. Write a Python program to remove item(s) from set """

                print("set:", obj.remove_set(set1))

            elif choice == 5:

                """ 5. Write a Python program to remove an item from a set if it is present in the set."""

                print("\n Original set :", set1 )

                print("\n After discard element", obj.discard_set(set1))

                print("\n ", set1)

            elif choice == 6:
                """ 6. Write a Python program to create an intersection of sets. """

                set6, set8 = obj.create_set1()
                print(" \n\n Intersection of two sets", obj.intersection_set(set6, set8))

            elif choice == 7:

                """ 7. Write a Python program to create a union of sets."""

                print("\n Union of two sets", obj.union_set(set6, set8))

            elif choice == 8:

                """ 8. Write a Python program to create set difference."""

                print("\n set Difference", obj.diff_set(set6, set8))

            elif choice == 9:

                """ 9. Write a Python program to create a symmetric difference."""

                print("\n set symmetric Difference", obj.sys_diff(set6, set8))

            elif choice == 10:

                """ 10. Write a Python program to clear a set."""

                print("set ", obj.clr_set(set1))

            elif choice == 11:

                """ 11. Write a Python program to use of frozen sets."""

                print("\n\nset frozen:", obj.frozen_set())

            elif choice == 12:

                """ 12. Write a Python program to find maximum and the minimum value in a set."""

                set1 = {45, 66, 88, 34}

                set13 = obj.max_set(set1)
                set14 = obj.min_set(set1)

                print("\n Maximum value in set", set13)
                print("\n Minimum value in set", set14)

            else:
                print("Enter valid choice Between 0-12")

        except Exception as e:
            print(e)