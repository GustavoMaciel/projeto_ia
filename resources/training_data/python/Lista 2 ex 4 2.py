num = []
for i in range(5):
    num.append(i*2)
print(num)

# num = [0, 2, 4, 6, 8]
num.insert(3,9)
print(num)

#num = [0, 2, 4, 9, 6, 8]

num.append(num.index(4))
print(num)

#num = [0, 2, 4, 9, 6, 8, 2]

del num[2]
print(num)

#num = [0, 2, 9, 6, 8, 2]

x = len(num) + 5

num.append(x)
print(num)

#num = [0, 2, 9, 6, 8, 6, 11]

num.sort()
print(num)

#num = [0, 2, 2, 6, 8, 9, 11]

num.remove(2)

#num = [0, 2, 6, 8, 9, 11]

print(num)
