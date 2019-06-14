# Python program to get the users environment.

import os
# Access all environment variables
print(os.environ)
# Access value of particular environment variables
print("XAUTHORITY VALUE",os.environ.get('XAUTHORITY'))
