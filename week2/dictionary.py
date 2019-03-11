
from week2.Utility.Util import Utility

class Dictionary:

    obj = Utility()

    dict_numbers = {'first': 1, 'second': 2, 'third': 3, 'Fourth': 4}

    dict5 = {'first  ': 1, 'second ': 2, 'third  ': 3, 'fourth ': 4, 'Fifth  ': 5, 'Sixth  ': 6, 'Seventh': 7}

    flag = True

    print("1. Write a Python script to sort (ascending and descending) a dictionary by value. ")
    print("2. Write a Python script to add a key to a dictionary. ")
    print("3. Write a Python script to concatenate following dictionaries to create a new one.")
    print("4. Write a Python program to iterate over dictionaries using for loops.")
    print("5. Write a Python script to generate dictionary a number (between 1 and n) in the form (x, x*x)")
    print("6. Write a Python program to remove a key from a dictionary. ")
    print("7. Write a Python program to print all unique values in a dictionary.")
    print("8. Write a Python program to create a dictionary from a string. ")
    print("9. Write a Python program to print a dictionary in table format.")
    print("10. Write a Python program to count the values associated with key in a dictionary. ")
    print("11. Write a Python program to convert a list into a nested dictionary of keys.")
    print("12. Write a Python program to check multiple keys exists in a dictionary")
    print("0. EXIT")

    while flag:

        try:

            print("___________________________________________________________________________________________________")

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            elif choice == 1:

                """1. Write a Python script to sort (ascending and descending) a dictionary by value. """

                print("Dictionary:", dict_numbers)

                print("\n Sort Dictionary:", obj.sort_dict(dict_numbers))

                print("\n Sort Dictionary reverse:", obj.sort_dict(dict_numbers))

            elif choice == 2:

                """2. Write a Python script to add a key to a dictionary. """

                print("\n\n Updated Dictionary:", obj.add_dict(dict_numbers))

                print("Updated Dictionary", dict_numbers)

            elif choice == 3:

                """3. Write a Python script to concatenate following dictionaries to create a new one. """

                dict1 = {'first': 1, 'second': 2, 'third': 3}
                dict2 = {'fourth': 4, 'Fifth': 5}
                dict3 = {'Sixth': 6, 'Seventh': 7}

                print("\n\n concatenate dictionary:", obj.concatenate_dict(dict1, dict2, dict3))

            elif choice == 4:

                """4. Write a Python program to iterate over dictionaries using for loops. """

                print("iterate over dictionary:", obj.iterate_dict(dict5))

            elif choice == 5:

                """5. Write a Python script to generate and print a dictionary that contains 
                    a number (between 1 and n) in the form (x, x*x)"""

                print("dictionary that contains a number in the form (x, x*x)", obj.create_dict())

            elif choice == 6:
                """6. Write a Python program to remove a key from a dictionary. """

                dict5 = {'first  ': 1, 'second ': 2, 'third  ': 3, 'fourth ': 4, 'Fifth  ': 5, 'Sixth  ': 6, 'Seventh': 7}

                # using Utility obj call del_dict()
                # return dictionary

                print("\n\n dictionary :", dict5)
                print("\n dictionary after delete key:", obj.del_dict(dict5))

            elif choice == 7:

                """7. Write a Python program to print all unique values in a dictionary. """

                dictC = [{"V": "S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII": "S005"}, {"V": "S009"}, {"VIII": "S007"}]

                # call unique_dict function
                # return dict
                print("\n\n unique", obj.unique_dict(dictC))

            elif choice == 8:

                """8. Write a Python program to create a dictionary from a string. 
                    Note: Track the count of the letters from the string.
                    Sample string : 'w3resource'
                    Expected output: {'3': 1, 's': 1, 'r': 2, 'u': 1, 'w': 1, 'c': 1, 'e': 2, 'o': 1}
                """

                dict1 = obj.str_validation()

                print("Dictionary from a string", obj.count_dict(dict1))

            elif choice == 9:

                """9. Write a Python program to print a dictionary in table format. """

                print(' Name     Number')

                for name, Number in dict5.items():
                    print('{}  :  {}'.format(name, Number))

            elif choice == 10:

                """10. Write a Python program to count the values associated with key in a dictionary. """

                # create dictionary

                student = [{'id': 1, 'success': True, 'name': 'Lary'},
                           {'id': 2, 'success': False, 'name': 'Rabi'},
                            {'id': 3, 'success': True, 'name': 'Alex'}]

                print("\n\n Count of how many dictionaries have success as True:", sum(d['success'] for d in student))

            elif choice == 11:

                """11. Write a Python program to convert a list into a nested dictionary of keys. """

                # list
                num_list = [1, 2, 3, 4]

                new_dict = current = {}

                for name in num_list:

                    current[name] = {}
                    current = current[name]

                print(new_dict)

            elif choice == 12:

                """12. Write a Python program to check multiple keys exists in a dictionary. """

                try:

                    if 'first  ' in dict5:

                        print("\n\n Yes 'first' key exists in dict")

                    else:

                        raise KeyError

                except KeyError:

                    print("\n\n No 'first' key does not exists in dict")

            elif choice == 13:

                """13. Write a Python program to count number of items in a dictionary value that is a list."""

                print("Length : %d" % len(dict5))

            else:

                print("Enter Valid choice between 0-13")

        except Exception as e:

            print(e)








