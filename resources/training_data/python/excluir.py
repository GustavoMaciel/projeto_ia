lista = []
pares = 0
qtdRodar = 5

for i in range(qtdRodar):
    num = lista.append(int(input("Entre com o número: ")))

for i in range(qtdRodar):
    if(lista[i] % 2 == 0):
        pares += 1

print("\nPares: %s\n" %pares)
print("%s\n"%lista)

for i in range(qtdRodar):
    print("%s" %lista[i])

for i in range(len(lista)):
    if(lista[i] == 2):
        print("\n2 está na posição: %s" %lista[i])
    
for i in "Dynamic":
    print(i)
