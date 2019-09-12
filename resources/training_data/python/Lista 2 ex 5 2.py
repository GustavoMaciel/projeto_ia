numAnimais = 2
animaisCadastrados = []

for i in range(numAnimais):
    nome = str.capitalize(input("Informe o nome do animal: "))
    idade = int(input("Informe a idade do animal: "))
    especie = str.capitalize(input("Informe a espécie do animal: "))
    animais = [nome, idade, especie]
    animaisCadastrados.append(animais)

print("\n")

busca = str.capitalize(input("Informe o nome do animal para busca: "))

for i in range(len(animaisCadastrados)):
    if(busca in animaisCadastrados[i][0]):
        print("Nome:", animaisCadastrados[i][0])
        print("Idade:", animaisCadastrados[i][1])
        print("Espécie:", animaisCadastrados[i][2])


