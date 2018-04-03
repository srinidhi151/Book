import sys

for line in sys.stdin:
    data = line.strip().split('\t')
    if len(data) != 7:
        continue

    #Analysis 1 :  Select the rows which has call duration of at least 15 mins
    print(line,end='')

    #Analysis 2:  Select the rows which has call type as ‘voice’ and call duration less than 10 mins
    if( int(data[3]) < 10):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 3: Select the rows which has call type as ‘sms’ and call duration more than 10mins
    if( int(data[3]) >= 10):
        for item in data:
            print(item,end='\t')
        print()

#Analysis 4:Select the calls that are before 12pm and call type as ‘voice’.
    time = data[2].strip('\'').split(':')
    if( int(time[0]) >= 12):
        for item in data:
            print(item,end='\t')
        print()

    #Analysis 5: Select the call that are between 12pm-1pm and call type as ‘sms’.
    if data[2].startswith('\'12:'):
        for item in data:
            print(item,end='\t')
        print()
