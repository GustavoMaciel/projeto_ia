la = [["Carol", 4], ["Pedro", 2], ["Laura", 3], ["Alice", 5], ["Luís", 4], ["Ludmila", 6]]
lb = [[7, 2, 5, 1], [9, 4, 3], [6, 8, 0, 2, 1], [4, 5, 10]]

"""A)"""
la[1][1] = 3


"""B)"""
subla = ["Bruno", 1]
la.append(subla)


"""C)"""
lb[2].append(7)


"""D)"""
for i in range(len(la)):
    if(la[i][1] < 4):
        print(la[i][0])


"""E)"""
for i in range(len(lb)):
    print(lb[i][0])
