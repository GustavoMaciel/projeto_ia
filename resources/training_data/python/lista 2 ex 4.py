amigos = 5
arrecadacao = 0
idosos = 80

for i in range (amigos):
    doacao = float(input("Entre com o valor doado: "))
    arrecadacao += doacao

arrecadamento = arrecadacao/idosos

print(arrecadamento)
