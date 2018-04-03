import sys

cur_num = ''
for line in sys.stdin:
    data = line.strip().split('\t')

    #Analysis 1: IMEI numbers for the latitude and longitude
    print(data[0])

    #Analysis 2: IMEI numbers for the call type voice and cell id = 5
    if data[1] == '\'5\'':
        print(data[0])

    #Analysis 3: IMSI for the given latitude and longitude
    if (cur_num != data[0]):
        print(data[0])
        cur_num = data[0]

    #Analysis 4: Subscriber phone numbers that belong to a particular latitude and longitude
    if (cur_num != data[0]):
        print(data[0])
        cur_num = data[0]
