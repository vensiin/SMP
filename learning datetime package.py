import datetime as dt # https://docs.python.org/3/library/datetime.html#


x = dt.datetime.strptime("31/01/22 23:59:59.999999",
                  "%d/%m/%y %H:%M:%S.%f")

x2 = dt.datetime.strftime(dt.datetime(2018, 6, 1),"%a %d %b %Y, %I:%M%p")

print(x2)

