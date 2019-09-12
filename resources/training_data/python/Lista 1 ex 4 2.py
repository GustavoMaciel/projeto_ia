lista = []
lista2 = []
desejado = 5
m = 0
for i in range(desejado):
    num = lista.append(int(input("Entre com o número: ")))

for i in range(len(lista)-1):
    num = lista2.append((lista[i] + lista[i+1]))
    
lista2.append(lista[-1])

print(lista)
print(lista2)
