# Python program to convert an integer to binary keep leading zeros.


# integer
x = 12
# method returns a formatted representation of the given value controlled by the format specifier
# value - value that needs to be formatted
# format_spec - The specification on how the value should be formatted.
print(format(x, '08b'))
# returns formatted value (octal)
print(format(x, '010b'))