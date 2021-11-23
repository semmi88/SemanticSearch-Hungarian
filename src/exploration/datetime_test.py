from datetime import datetime

dt = datetime.now()
print(dt)
print(dt.strftime('%a %d-%m-%Y'))
print(dt.strftime('%a %d/%m/%Y'))
print(dt.strftime('%a %d/%m/%y'))
print(dt.strftime('%A %d-%m-%Y, %H:%M:%S'))
print(dt.strftime('%X %x'))
print(dt.strftime('%Y-%m-%d_%H:%M:%S'))
