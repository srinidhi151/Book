import sys

for line in sys.stdin:
    data = line.strip().lower().split(',')
    if len(data) != 7:
        continue
    print(data[4]+'\t'+data[5])
