# Python program to get the name of the host on which the routine is running.

# for socket
import socket
# Get local machine name
hostname = socket.gethostname()
# Get local machine ip addr
IPAddr = socket.gethostbyname(hostname)
# printing hostname
print("Your Computer Name is:" + hostname)
# printing Ip address
print("Your Computer IP Address is:" + IPAddr)
