
from week2.Utility.Util import Utility
import textwrap

class String:

    obj = Utility()
    flag = True

    print("1. Write a Python program to calculate the length of a string.  ")
    print("2. Write a Python program to count the number of characters (character frequency) in a string.   ")
    print("3. Python program str occurrences of its first char have been changed to '$', except the first char itself.")
    print("4. Python program to add 'ing' at the end of a given str. If ends with 'ing' then add 'ly' instead")
    print("5. Write a Python function that takes a list of words and returns the length of the longest one.")
    print("6. Python script that takes input from the user and displays that input back in upper and lower cases. ")
    print("7. program accepts a comma separated sequence of words as input and prints the unique words in sorted form ")
    print("8. Write a Python program to get the last part of a string before a specified character.  ")
    print("9. Write a Python program to display formatted text (width=50) as output. ")
    print("10. Write a Python program to count occurrences of a substring in a string.  ")
    print("11. Write a Python program to reverse a string. ")
    print("12. Write a Python program to lowercase first n characters in a string. ")
    print("0. EXIT")

    while flag:

        try:

            print("___________________________________________________________________________________________________")

            choice = int(input("Enter your choice"))

            if choice == 0:
                flag = False

            elif choice == 1:

                """1. Write a Python program to calculate the length of a string. """

                # call str validation () in Utility
                print(obj.str_validation())

            elif choice == 2:

                """2. Write a Python program to count the number of characters (character frequency) in a string.  """

                # call str validation fun
                str2 = obj.str_validation()

                # call count char fun
                str1 = obj.count_char(str2)

            elif choice == 3:

                """3. Write a Python program to get a string from a given string where all occurrences of
                 its first char have been changed to '$', except the first char itself.  """

                str2 = obj.str_validation()
                print("\n\nyour Enter string", str2)
                str3 = obj.change_char(str2)
                print("\n\nAfter changed", str3)

            elif choice == 4:

                """4. Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
                 If the given string already ends with 'ing' then add 'ly' instead. 
                 If the string length of the given string is less than 3, leave it unchanged."""

                print(obj.strin_ing())

            elif choice == 5:

                """5. Write a Python function that takes a list of words and returns the length of the longest one.  """

                try:
                    list_words = []
                    num = input("Enter the number of elements in list:")
                    num1 = int(num)
                    # print(type(num1))
                    if num.isdigit():
                        pass

                        try:

                            for temp in range(0, num1):

                                string = input("Enter element" + str(temp + 1) + ":")

                                if string.isalpha():

                                    list_words.append(string)

                                    s1 = obj.longest_words(list_words)
                                    # print("longest words", s1)

                                else:

                                    raise Exception
                            print("longest words", s1)

                        except Exception as e:

                            print("enter string only", e)
                    else:
                        raise Exception

                except Exception as e:

                    print("enter digit only", e)

                    try:
                        print("The word with the longest length is:", s1)

                    except NameError:
                        print("Enter valid string")

            elif choice == 6:

                """6. Python script that takes input  displays that input back in upper and lower cases."""

                s1 = obj.str_validation()

                s2 = obj.upper_str(s1)

                print("Upper Case ", s2)

                s3 = obj.lower_str(s1)

                print("Lower Case", s3)

            elif choice == 7:

                """7. Write a Python program that accepts a comma separated sequence of words as input and 
                    prints the unique words in sorted form (alphanumerically)."""

                print("Enter a  separated sequence of words:")

                lst = [n for n in input().split(',')]

                lst.sort()

                print("Sorted:")

                print(','.join(lst))

            elif choice == 8:

                """8. Write a Python program to get the last part of a string before a specified character. """

                str1 = 'https://www.w3resource.com/python-exercises/string'
                print(str1.rsplit('/', 1)[0])
                print(str1.rsplit('-', 1)[0])

            elif choice == 9:

                """9. Write a Python program to display formatted text (width=50) as output."""

                value = """This function wraps the input paragraph such that each line in the paragraph is at most width
                        characters long. The wrap method returns a list of output lines.The returned list is empty if 
                        the wrapped output has no content."""

                # Wrap this text.
                wrapper = textwrap.TextWrapper(width=50)

                word_list = wrapper.wrap(text=value)

                # Print each line.

                for element in word_list:
                    print(element)

            elif choice == 10:

                """10. Write a Python program to count occurrences of a substring in a string. """

                string = "Python is awesome, isn't it?"
                substring = "i"

                # count after first 'i' and before the last 'i'
                count = string.count(substring, 8, 25)
                count = string.count(substring)

                # print count
                print("The count is:", count)

            elif choice == 11:

                """11. Write a Python program to reverse a string."""

                str2 = obj.str_validation()

                print("\n The original string is : ", str2)

                str3 = obj.reverse_string(str2)

                print("\n The reversed string(using is : ", str3)

            elif choice == 12:

                """12. Write a Python program to lowercase first n characters in a string. """

                str1 = 'Write a Python script'
                print(str1[:4].lower() + str1[4:])

            else:
                print("Enter Valid choice between 0-12")

        except Exception as e:

            print(e)