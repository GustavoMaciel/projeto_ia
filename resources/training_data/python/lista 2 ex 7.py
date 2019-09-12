semana = 7
cont = 0
meta = 100
soma = 0

for i in range(semana):
    cartas = int(input("Entre com a quantidade de cartas entregue: "))
    if(cartas >= meta):
        superior = cartas
        cont += 1
    soma += cartas

media = soma//semana

print("\nCumpriu a meta em",cont,"dias.")
print("Média de entregas:",media)
        
