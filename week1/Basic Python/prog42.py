# Python program to determine if the python shell is executing in 32bit or 64bit mode on operating system.

# struct module can be used in handling binary data stored in files, database or from network connections
import struct
# Return the size of the struct (and hence of the string) corresponding to the given format
print(struct.calcsize("P") * 8)