menor = 100000000000000000
lista = []
desejados = 5

for i in range(desejados):
    num = int(input("Entre com o número: "))
    lista.append(num)
    if(num < menor):
        menor = num

print("O menor é: %s"%menor)
