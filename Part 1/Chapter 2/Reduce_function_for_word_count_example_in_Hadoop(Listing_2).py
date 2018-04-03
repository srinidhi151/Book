import sys

cur_count = 0
cur_word = None

for line in sys.stdin:
    word = line.strip().lower()
    if( cur_word != None and cur_word != word):
        print("{0}\t{1}".format(cur_word,cur_count))
        cur_count = 0

    cur_word = word
    cur_count = cur_count + 1

if( cur_word != None):
    print("{0}\t{1}".format(cur_word,cur_count))
