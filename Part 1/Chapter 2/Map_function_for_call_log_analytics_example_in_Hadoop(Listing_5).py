import sys

for line in sys.stdin:
    data = line.strip().split(',')
    if len(data) != 7:
        continue

    #Analysis 1: Select the rows which has call duration of at least 15 mins
    if( int(data[3]) >= 15):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 2:  Select the rows which has call type as ‘voice’ and call duration less than 10 mins
    if( data[6] == '\'voice\'' ):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 3: Select the rows which has call type as ‘sms’ and call duration more than 10mins
    if( data[6] == '\'sms\'' ):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 4:Select the calls that are before 12pm and call type as ‘voice’.
    if( data[6] == '\'voice\'' ):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 5: Select the call that are between 12pm-1pm and call type as ‘sms’.
    if( data[6] == '\'sms\'' ):
        for item in data:
            print(item,end='\t')
        print()
