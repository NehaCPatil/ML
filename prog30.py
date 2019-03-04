# Python program to access and print a URL's content to the console.

# Used to make requests
import urllib.request
# urlopen with this Request object returns a response object for the URL requested
b = urllib.request.urlopen("https://www.youtube.com/user/guru99com")
# print the responce object for the URL requested
print("result code:"+str(b.getcode()))
a = b.read()
print(a)



