# python program to access environment variables.

import os
# Environment variables are accessed through os.environ
print(os.environ['HOME'])
# list of all the environment variables
print(os.environ)
# using get will return `None` if a key is not present rather than raise a `KeyError`
print(os.environ.get('KEY_THAT_MIGHT_EXIST'))
print(os.environ.get('HOME'))