lista = []
desejados = 8
multiplos = []


for i in range(desejados):
    num = lista.append(int(input("Entre com o número desejado: ")))
    if(lista[i] % 3 == 0):
        multiplos.append(lista[i])

for i in range(len(multiplos)):
    print(multiplos[i])
