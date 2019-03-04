#  Python program to find the available built-in modules


# sys module provides allows us to operate on underlying interpreter
import sys
# for loop for find all builtin module name
for name in sys.builtin_module_names:
    # printing built in module name
    print(name)
