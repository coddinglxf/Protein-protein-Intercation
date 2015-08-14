__author__ = 'Administrator'
# coding=-utf-8_

f_2 = open('newTF.txt', 'r')

dic = dict()
index = 1
for each in f_2:
    words = str(each).lstrip().rstrip().split(" ")
    for word in words:
        if word not in dic:
            dic[word] = index
            index += 1
f_2.close()
print dic

f = open('newTF.txt', 'r')
w = open("keda", 'w')
for each in f:
    temp = str()
    word = str(each).lstrip().rstrip().split(" ")
    if word[0] is "1":
        temp = "false" + "@"
    else:
        temp = "true" + "@"
    for i in xrange(1, len(word)):
        temp = temp + str(dic[word[i]]) + "@"
    temp = temp[0:len(temp) - 1]
    w.write(temp + "\n\r")
w.close()
f.close()
