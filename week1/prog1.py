""" Python program accepts the user's first and last name and print them in reverse order with a space between them."""

try:
    fname = str(input("Enter your first name"))
    lname = str(input("Enter your last name"))

    if fname.isalpha() and lname.isalpha():
         print(fname)
         print(lname)
         print(lname, fname)
    else:
        raise TypeError
except TypeError:
    print("Letter only Please")
