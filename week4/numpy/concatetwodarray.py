from week4.numpy.Utility.util import Utility


# class

class Array:

    # constructor

    def __init__(self):
        self.obj = Utility()

    
    # function for display concatenate array

    def display_concatenate(self):
        # call method
        print(self.obj.concatenate_array())

    # function for display immutable array

    def display_array_immutable(self):
        # call method
        self.obj.array_immutable()

    # function for display multiply every element array

    def display_multiply_every_element(self):
        # call method
        self.obj.multiply_every_element()

    # function for display list

    def display_arr_to_list(self):
        # call method
        print(self.obj.array_to_list())

    # function for display scientific notation

    def display_scientific_notation(self):
        # call method
        self.obj.scientific_notation()

    # function for display extra column

    def display_extra_column(self):
        # call method
        self.obj.extra_column()

    # function for display delete element

    def display_delete_element(self):
        # call method
        print("Element", self.obj.delete_element())


# problem statement
print("_______________________________________________________________________________________________________________")
print("20. Write a Python program to concatenate two 2-dimensional arrays. ")
print("21. Write a Python program to make an array immutable (read-only). ")
print("22. create an array of (3, 4) shape, multiply every element value by 3 and display the new array.  ")
print("23. Write a Python program to convert a NumPy array into Python list structure. ")
print("24. Write a Python program to suppresses the use of scientific notation for small numbers in numpy array. ")
print("25. Write a Python program to how to add an extra column to an numpy array.  ")
print("27. Write a Python program to remove specific elements in a numpy array. ")

# object of class

obj = Array()

# while loop
while True:

    #
    try:

        ch = input("\n\nEnter choice")

        choice = int(ch)

        if ch.isdigit():

            if choice == 1:
                obj.display_concatenate()

            elif choice == 2:
                obj.display_array_immutable()

            elif choice == 3:
                obj.display_multiply_every_element()

            elif choice == 4:
                obj.display_arr_to_list()

            elif choice == 5:
                obj.display_scientific_notation()

            elif choice == 6:
                obj.display_extra_column()

            elif choice == 7:
                obj.display_delete_element()

            elif choice == 8:
                exit()

        else:
            raise Exception

    except Exception:
        print("\n Invalid Input")
