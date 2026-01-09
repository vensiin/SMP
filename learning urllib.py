import urllib.request  # https://docs.python.org/3/library/urllib.request.html#module-urllib.request

with urllib.request.urlopen("https://docs.python.org/3/howto/urllib2.html") as response:
    html = response.read().decode("utf-8")
print(html[:10000])

