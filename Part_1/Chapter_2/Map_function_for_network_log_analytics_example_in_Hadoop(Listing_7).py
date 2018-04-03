import sys

for line in sys.stdin:
    data = line.strip().split(',')
    if len(data) != 8:
        continue

    #Analysis 1: IMEI numbers for the latitude and longitude
    lat = '\'16.5N\''
    lon = '\'56.4S\''
    if (data[6] == lat and data[7] == lon):
        print(data[2])

    #Analysis 2: IMEI numbers for the call type voice and cell id = 5
    if (data[3] == '\'voice\''):
        print("{0}\t{1}".format(data[2],data[4]))

    #Analysis 3: IMSI for the given latitude and longitude
    lat = '\'46.5N\''
    lon = '\'55.4S\''
    if (data[6] == lat and data[7] == lon):
        print(data[1])

    #Analysis 4: Subscriber phone numbers that belong to a particular latitude and longitude
    lat = '\'76.5N\''
    lon = '\'56.4S\''
    if (data[6] == lat and data[7] == lon):
        print(data[5])
