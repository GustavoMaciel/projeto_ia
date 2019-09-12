superior = 0
bim = 6
cont = 0

for i in range(bim):
    nota = float(input("Entre com a nota: "))
    
    if(nota > superior):
        superior = nota
        cont += 1

if(cont == bim):
    print("Ganhara brinquedo")

else:
    print("Nao ganhara brinquedo")
