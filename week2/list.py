from week2.Utility.Util import Utility
from itertools import groupby
from operator import itemgetter
import itertools


class List:
    obj = Utility()

    list1 = [1, 3, 5, 7, 6, 4, 1, 4]

    flag = True

    print("1. Write a Python program to sum all the items in a list. ")
    print("2. Write a Python program to multiplies all the items in a list. ")
    print("3. Write a Python program to get the smallest number from a list. ")
    print("4. Python program to count the number of str the first and last character are same from a given list of str")
    print("5. Write a Python program to get a list, sorted in increasing order by the last element")
    print("6. Write a Python program to remove duplicates from a list.  ")
    print("7. Write a Python program to clone or copy a list. ")
    print("8. Python function that takes two lists and returns True if they have at least one common member")
    print("9. Write a Python program to print a specified list after removing the 0th, 4th and 5th elements.")
    print("10. Write a Python program to generate all permutations of a list in Python.")
    print("11. Write a Python program to get the difference between the two lists.")
    print("12. Write a Python program to append a list to the second list. ")
    print("13. Write a python program to check whether two lists are circularly identical.")
    print("14. Write a Python program to find common items from two lists.")
    print("15. Write a Python program to split a list based on first character of word.")
    print("16. Write a Python program to remove duplicates from a list of lists.  ")
    print("0. EXIT")

    while flag:

        try:

            print('___________________________________________________________________________________________________')

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            if choice == 1:

                """ 1. Write a Python program to sum all the items in a list.  """
                try:
                    # create empty list
                    list_words = []

                    # get num of element from user
                    number = input("\n\nEnter the number of elements in list:")
                    num1 = int(number)

                    # if user enter number is valid
                    if number.isdigit():

                        # enter element in list
                        input_string = input("Enter a list element separated by space ")

                        # for space
                        list11 = input_string.split()

                        sum_element = 0

                        # for all num in list
                        for num in list11:
                            # sum of element
                            sum_element += int(num)

                        if num.isdigit():

                            print("Sum = ", sum_element)

                        else:
                            raise Exception
                    else:
                        raise Exception

                except Exception as e:

                    print("enter digit only", e)

            elif choice == 2:

                """2. Write a Python program to multiplies all the items in a list.  """

                print("multiplies all the items in a list:", obj.multi_list(list1))

            elif choice == 3:

                """3. Write a Python program to get the smallest number from a list.  """

                print("smallest number from a list", obj.min_list(list1))

            elif choice == 4:

                """4. Write a Python program to count the number of strings where the string length is 2 or more and the
                first and last character are same from a given list of strings.   
                Sample List : ['abc', 'xyz', 'aba', '1221']
                Expected Result : 2"""

                words_list = ['abc', 'xyz', 'aba', '1221']

                print("\n\n first and last character same from list of strings count: ", obj.match_words(words_list))

            elif choice == 5:

                """5. Write a Python program to get a list, sorted in increasing order by the last element in each tuple from a given list of non-empty tuples.   
                Sample List : [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
                Expected Result : [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)]"""

                Sample_List = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
                print("Sorted:", obj.sort(Sample_List))

            elif choice == 6:

                """6. Write a Python program to remove duplicates from a list. """

                print("\n\n my result list:", obj.repet_list(list1))

            elif choice == 7:

                """7. Write a Python program to clone or copy a list. """

                print("\n\n clone or copy a list:", obj.cloning_list(list1))

            elif choice == 8:

                """8. Python function that takes two lists and returns True if they have at least one common member."""

                a_set = [1, 2, 3, 4, 5]
                b_set = [5, 6, 7, 8, 9]

                print("\n\n Common member", obj.common_member(a_set, b_set))

                a_set = [1, 2, 3, 4, 5]
                b_set = [6, 7, 8, 9]

                print("\ncommon member", obj.common_member(a_set, b_set))

            elif choice == 9:

                """9. Write a Python program to print a specified list after removing the 0th, 4th and 5th elements. """

                print("\n list after removing 0th, 4th and 5th elements:", obj.remove_specific_element(list1))

            elif choice == 10:

                """10. Write a Python program to generate all permutations of a list in Python. """

                list_1 = [1, 2, 3, 4, 5]

                print("permutations of a list in Python", obj.permute(list_1))

            elif choice == 11:

                """11. Write a Python program to get the difference between the two lists."""

                list_A = [10, 15, 20, 25, 30, 35, 40]
                list_B = [25, 40, 35]

                print("difference between the two lists:", obj.diff_list(list_A, list_B))

            elif choice == 12:

                """12. Write a Python program to append a list to the second list. """

                # Initializing lists
                list1 = [1, 4, 5, 6, 5]
                list2 = [3, 5, 7, 2, 5]

                print("list:", obj.append_list(list1, list2))

            elif choice == 13:

                """13. Write a python program to check whether two lists are circularly identical.  """

                list1 = [10, 10, 0, 0, 10]
                list2 = [10, 10, 10, 0, 0]
                list3 = [1, 10, 10, 0, 0]

                # check for list 1 and list 2
                if obj.circularly_identical(list1, list2):
                    print("Yes")
                else:
                    print("No")

                # check for list 2 and list 3
                if obj.circularly_identical(list2, list3):
                    print("Yes")
                else:
                    print("No")

            elif choice == 14:

                """14. Write a Python program to find common items from two lists.  """

                a = [1, 2, 3, 4, 5]
                b = [5, 6, 7, 8, 9]

                print("common items from two lists", obj.common_member(a, b))

                a = [1, 2, 3, 4, 5]
                b = [6, 7, 8, 9]

                print("common items from two lists", obj.common_member(a, b))

            elif choice == 15:

                """15. Write a Python program to split a list based on first character of word.  """

                word_list = ['give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'leave', 'call']

                for letter, words in groupby(sorted(word_list), key=itemgetter(0)):

                    print("\n\n letter:-", letter)

                    for word in words:
                        print("\n word:-", word)

            elif choice == 16:

                """16. Write a Python program to remove duplicates from a list of lists."""

                num = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]

                print("Original List:-", num)
                num.sort()

                new_num = list(num for num, _ in itertools.groupby(num))

                print("New List:-", new_num)

            else:
                print("Enter Valid choice between 0-16")

        except Exception as e:

            print(e)
