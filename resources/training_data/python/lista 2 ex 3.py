num1 = int(input("Entre com seu primeiro número: "))
num2 = int(input("Entre com seu segundo número: "))
multiplos = 0

if(num2>num1):
    inicial = num1+1
    limite = num2-1
else:
    inicial = num2+1
    limite = num1-1
    
for cont in range(inicial,limite):
    if(cont % 4 == 0):
        multiplos += 1

print(multiplos,"múltiplos.")
