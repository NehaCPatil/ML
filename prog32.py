# Python program to get the effective group id, effective user id, real group id, a
# list of supplemental group ids associated with the current process.


import os
# Return the effective group id of the current process
print("\nEffective group id: ", os.getegid())
# Return the current process’s effective user id
print("Effective user id: ", os.geteuid())
# Return the real group id of the current process
print("Real group id: ", os.getgid())
# Return list of supplemental group ids associated with the current process
print("List of supplemental group ids: ", os.getgroups())
print()
