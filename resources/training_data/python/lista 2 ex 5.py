def tratarErroFloat():
    while True:
        try:
            salario = float(input("Entre com o salário: "))
            break
        except ValueError:
            print("Somente números flutuantes.")

def tratarErroFloat2():
    while True:
        try:
            despesas = float(input("Entre com as despesas do mês: "))
            break
        except ValueError:
            print("Somente números flutuantes.")

tratarErroFloat()

ano = 3
economizado = 0

for i in range(ano):
    tratarErroFloat2()
    restos = salario - despesas
    economizado += restos

media = economizado/ano

print(media)
