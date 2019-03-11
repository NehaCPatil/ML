from array import *
import textwrap


class Utility:

    # (__________________________________________ARRAY_________________________________________________________________)

    # Display array

    def display_arr(self, arr):

        for _ in arr:
            # return all element in array
            return arr

    """ 2. Write a Python program to reverse the order of the items in the array. """

    # Reverse Array
    def arr_revers(self, arr):
        arr.reverse()
        for temp in range(len(arr)):
            return arr[temp]

    """ 3. Write a Python program to get the number of occurrences of a specified element in an array.  """

    # Count
    def arr_count(self, arr):
        temp = arr.count(30)
        return temp

    """ 4. Write a Python program to remove the first occurrence of a specified element from an array. """

    # Remove Element
    def arr_remove(self, arr):
        arr.remove(30)
        return arr

    # (__________________________________________SET___________________________________________________________________)

    """2. Write a Python program to iteration over sets. """

    # display set
    def display_set(self, set1):
        for temp in set1:
            print(temp)

    """3. Write a Python program to add member(s) in a set."""

    # add set
    def add_set(self, temp):
        temp.update([1, 6, 8, 9, 2])
        return temp

    """4. Write a Python program to remove item(s) from set"""

    # remove set
    def remove_set(self, set1):
        try:
            set1.remove(45)

        except KeyError:
            return "enter valid index"

        finally:
            return set1

    """5. Write a Python program to remove an item from a set if it is present in the set. """

    # discar set
    def discard_set(self, set1):
        set1.discard(55)

    """6. Write a Python program to create an intersection of sets. """

    # create set
    def create_set1(self):
        # Creating two sets
        setA = set([1, 2, 3, 4, 5])
        setB = set([0, 2, 4, 6, 8])
        return setA, setB

    # intersection set
    def intersection_set(self, set6, set8):
        setC = set6.intersection(set8)
        return setC

    """7. Write a Python program to create a union of sets"""

    # union set
    def union_set(self, set6, set8):
        setD = (set6 | set8)
        return setD

    """8. Write a Python program to create set difference. """

    # set difference
    def diff_set(self, set6, set8):
        setE = set6.difference(set8)
        return setE

    """9. Write a Python program to create a symmetric difference. """

    # symmetric diff
    def sys_diff(self, setA, setB):
        setA = {'a', 'b', 'c', 'd'}
        setB = {'c', 'd', 'e'}
        setC = setA.symmetric_difference(setB)
        setD = setB.symmetric_difference(setA)
        return setC, setD

    """10. Write a Python program to clear a set. """

    # clear set
    def clr_set(self, set1):
        set1.clear()

    """11. Write a Python program to use of frozen sets. """

    # frozen set
    def frozen_set(self):
        vowels = ('a', 'e', 'i', 'o', 'u')
        setF = frozenset(vowels)
        return setF

    """12. Write a Python program to find maximum and the minimum value in a set. """

    # maximum value
    def max_set(self, set1):
        setF = max(set1)
        return setF

    # minimum value
    def min_set(self, set1):
        setG = min(set1)
        return setG

    # (_______________________________________STRING__________________________________________________________________)

    """ 1. Write a Python program to calculate the length of a string."""

    def str_validation(self):
        try:
            string = input("Enter string")

            if string.isalpha():
                print("length of string", len(string))
                print(string)
                return string
            else:
                raise ValueError
        except ValueError:
            print("Letter only Please")

    """ 2. Write a Python program to count the number of characters (character frequency) in a string.  """

    def count_char(self, string):
        res = {}
        for keys in string:
            res[keys] = res.get(keys, 0) + 1
        print(" Count of all characters in  is : \n" + str(res))

    """3. Write a Python program to get a string from a given string where all occurrences of
                    its first char have been changed to '$', except the first char itself.  """

    def change_char(self, string):
        char = string[0]
        string = string.replace(char, '$')
        string = char + string[1:]
        return string

    """4. Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
                 If the given string already ends with 'ing' then add 'ly' instead. 
                 If the string length of the given string is less than 3, leave it unchanged."""

    def isConSpace(self, string):
        for temp in string:
            if temp == " ":
                return True
        return False

    def strin_ing(self):

        string = input("Enter a string : ")

        if len(string) > 0:

            if self.isConSpace(string) == False:

                if string.isalpha():

                    if len(string) > 2:

                        if string[-3:] == "ing":
                            string += "ly"
                            print(string)
                        else:
                            string += "ing"
                            print(string)
                    else:
                        print(string)
                else:
                    print("String must be contain only alphabet.")
            else:
                print("Please enter a string without space.")
        else:
            print("String is empty.")

    """5. Write a Python function that takes a list of words and returns the length of the longest one.  """

    def longest_words(self, list_words):

        max1 = len(list_words[0])
        temp1 = list_words[0]
        for temp2 in list_words:
            if len(temp2) > max1:
                max1 = len(temp2)
                temp1 = temp2
        return temp1

    """6. Python script that takes input  displays that input back in upper and lower cases."""

    def upper_str(self, string):
        s1 = string.upper()
        return s1

    def lower_str(self, string):
        s2 = string.lower()
        return s2

    def wrapper_text(self, value):
        # Wrap this text.
        wrapper = textwrap.TextWrapper(width=50)

        word_list = wrapper.wrap(text=value)

        # Print each line.

        for element in word_list:
            return element

    """11. Write a Python program to reverse a string."""

    def reverse_string(self, str2):
        str1 = ""
        for i in str2:
            str1 = i + str1
        return str1

    # (_____________________________TUPLE______________________________________________________________________________)
    # create tuple

    def tuple_create(self):
        thistuple = ("apple", "banana", "cherry")
        return thistuple

    # create a tuple with different data types

    def tuple_diff(self):
        thistuple = ("python", " tutorial", 1, " phy", 99)
        return thistuple

    # colon of a tuple

    def Cloning(self, li1):
        li_copy = []
        li_copy.extend(li1)
        return li_copy

    # repeated items of a tuple

    def repet_element(self, vowels):
        list1 = []
        list2 = list(vowels)
        for var in list2:
            if list2.count(var) > 1:
                list1.append(var)

        set1 = set(list1)
        return set1, list2

    # reverse a tuple

    def Reverse_tuple(self, number):
        new_tup = number[::-1]
        return new_tup

    # (________________________________________DICTIONARY_____________________________________________________________)

    def sort_dict(self, dict_numbers):
        s_dict = sorted(dict_numbers.values())
        r_sort = sorted(dict_numbers.values(), reverse=True)
        return s_dict, r_sort

    def add_dict(self, dict_numbers):
        a_dict = dict_numbers["Fifth"] = 5
        return a_dict

    def concatenate_dict(self, dict1, dict2, dict3):
        for _ in (dict1, dict2, dict3):
            dict4 = dict(dict1)
            dict4.update(dict2)
            dict4.update(dict3)
        return dict4

    def iterate_dict(self, dict5):
        for key, value in dict5.items():
            print("\n key:%s" % key)
            print(" value:%s" % value)

    def create_dict(self):
        try:
            num = input("Enter a num")
            num1 = int(num)
            if num.isdigit():
                dict1 = {x: x * x for x in range(1, num1 + 1)}
                return dict1
            else:
                raise ValueError

        except ValueError:
            print("Enter number only")

    def del_dict(self, dict5):
        try:
            del dict5['first  ']
            return dict5
        except KeyError:
            print("Invalid Key or key not present in dictionary")

    def unique_dict(self, dictC):
        unique = set(val for dic in dictC for val in dic.values())
        # for element in unique:
        return unique

    def count_dict(self, string):
        res = {}
        for keys in string:
            res[keys] = res.get(keys, 0) + 1
        print(" Count of all characters in  is : \n" + str(res))

    # ("___________________________________LIST_________________________________________________________________________")

    # multiplies all the items in a list

    def multi_list(self, list1):
        my_new_list = [i * 5 for i in list1]
        return my_new_list

    # smallest number from a list

    def min_list(self, list1):
        list2 = min(list1)
        return list2

    # first and last character are same from a given list of strings

    def match_words(self, words_list):
        count = 0

        for word in words_list:
            if len(word) > 1 and word[0] == word[-1]:
                count += 1
        return count

    def last(self, n):
        return n[-1]

    def sort(self, Sample_List):
        return sorted(Sample_List, key=self.last)

    # element remove
    def repet_list(self, my_list):
        list1 = []
        # list2 = list(my_list)
        for var in my_list:
            if my_list.count(var) > 1:
                list1.append(var)

        set1 = set(my_list)
        return set1

    # cloning

    def cloning_list(self, list1):
        list_copy = list1[:]
        return list_copy

    # commons member

    def common_member(self, a_set, b_set):
        set_a = set(a_set)
        set_b = set(b_set)
        if set_a & set_b:
            return True
        else:
            return False

    # remove specific element

    def remove_specific_element(self, list1):
        my_list = [x for (i, x) in enumerate(list1) if i not in (0, 4, 5)]
        return my_list

    # permutations

    def permute(self, list_1):
        num = len(list_1)
        result = []
        temp1 = num * [0]

        result.append(list_1)

        i = 0
        while i < num:
            if temp1[i] < i:
                if i % 2 == 0:
                    tmp = list_1[0]
                    list_1[0] = list_1[i]
                    list_1[i] = tmp

                else:

                    tmp = list_1[temp1[i]]
                    list_1[temp1[i]] = list_1[i]
                    list_1[i] = tmp

                result.append(list_1)
                temp1[i] += 1
                i = 0
            else:
                temp1[i] = 0
                i += 1

        return result

    # difference between the two lists

    def diff_list(self, list_A, list_B):
        return list(set(list_A) - set(list_B))

    # append a list to the second list

    def append_list(self, list1, list2):
        my_list = list2 + list1
        return my_list

    # two lists are circularly identical

    def circularly_identical(self, list1, list2):
        return ' '.join(map(str, list2)) in ' '.join(map(str, list1 * 2))

    def common_member(self, a, b):
        a_set = set(a)
        b_set = set(b)
        # check length

        if len(a_set.intersection(b_set)) > 0:
            return a_set.intersection(b_set)
        else:
            return "no common elements"
