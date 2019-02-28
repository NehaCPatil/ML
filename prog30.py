import urllib.request
b = urllib.request.urlopen("https://www.youtube.com/user/guru99com")
print("result code:"+str(b.getcode()))
a = b.read()
print(a)


# from http.client import HTTPConnection
# conn = HTTPConnection("example.com")
# conn.request("GET", "/")
# result = conn.getresponse()
# # retrieves the entire contents.
# contents = result.read()
# print(contents)
