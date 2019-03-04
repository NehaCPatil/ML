# Python program accepts the user's first and last name and print them in reverse order with a space between them.

# getting user input
f_name = input("Enter first name:")
l_name = input("Enter Last name:")
# Creating a list
list_of_l_name = list(l_name)
list_of_f_name = list(f_name)
# joined by separator
reverse_f_name = ''.join(list_of_f_name)
reverse_l_name = ''.join(list_of_l_name)
# print list
print(reverse_l_name + " " + reverse_f_name)
