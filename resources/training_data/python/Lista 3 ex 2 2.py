mes = []
receita = []
despesa = []
ano = 4
somaReceita = 0
somaDespesas = 0
superar = []

orcamento = []

for i in range(ano):
    x = mes.append(input("Mês: "))
    y = receita.append(float(input("Receita: ")))
    z = despesa.append(float(input("Despesa: ")))
    
    whatever = [x,y,z]
    orcamento.append(whatever)
    
for i in range(ano):
    somaReceita += receita[i]
    somaDespesas += despesa[i]
    if(receita[i] > despesa[i]):
        superar.append(mes[i])
        
saldo = somaReceita - somaDespesas

if(saldo >= 0):
    print("O saldo financeiro foi positivo.")
else:
    print("O saldo foi negativo.")

print(end="A receita superou a despesa em: " + str(superar))

print("\n\n\n")

paraDespesa = 0
maiorDespesa = " "

for i in range(len(orcamento)):
    if(orcamento[i][1] > 2500):
        print(orcamento[i][0])
        
    if(orcamento[i][2] > paraDespesa):
        maiorDespesa = orcamento[i][0]

print("Maior despesa: "+ maiorDespesa)

receitaMedia = somaReceita / ano

print("Receita média: " str(receitaMedia))
