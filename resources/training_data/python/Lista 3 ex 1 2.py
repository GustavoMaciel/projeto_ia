quant = 5
Lista = []
for i in range(quant):
    
    x = i + 2
    Lista.append(x)
    #Lista = [2, 3, 4, 5, 6]
for i in range(quant):
    if (Lista[i] % 2 == 0):
        Lista[i] = Lista[i] + 3

    #Lista = [5,3,7,5,9]

print(Lista)

print(Lista[3]*2)

#10

print(Lista[0] - Lista[4])

#-4
