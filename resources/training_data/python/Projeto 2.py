nomeAluno = []
matriculaAluno = []
notaAluno1 = []
notaAluno2 = []
notaAluno3 = []

def menu():
    print("\nSigaa Docente \n")
    print("[1] Cadastrar aluno\n[2] Listar\n[3] Buscar\n[x] Sair\n")

def buscar():
    busca = input("Entre com a matrícula: ")
    while(busca not in matriculaAluno):
        print("Matrícula não cadastrada.")
        busca = input("Entre com a matrícula: ")
    while(busca.isdigit() == False ):
        print("Apenas números são aceitos.")
        busca = input("Entre com a matrícula: ")
    while(len(busca) < 10 or len(busca) > 10):
        print("A matrícula precisa ter dez números.")
        busca = input("Entre com a matrícula: ")
        cont = 0
    if(busca in matriculaAluno):
        for i in range(len(matriculaAluno)): 
            if(matriculaAluno[i] == busca):
                nome1 = nomeAluno[i]
                matricula1 = matriculaAluno[i]
                print("\nNome: %s"%nome1)
                print("Matrícula: %s"%matricula1)
                media1 = (notaAluno1[i] + notaAluno2[i] + notaAluno3[i])/3
                if(media1 >= 7):
                    print("Aluno aprovado com média: %.1f"%media1)
                else:
                    print("Aluno reprovado com média: %.1f"%media1)
def situacao():
    while True:
        try:
            nota1 = float(input("Entre com a nota: "))
            while(nota1 < 0 or nota1 > 10):
                print("Entrada inválida.")
                nota1 = float(input("Entre com a nota: "))
            notaAluno1.append(nota1)
            break
        except ValueError:
            print("Só são aceitos números.")
        
    while True:
        try:
            nota1 = float(input("Entre com a nota: "))
            while(nota1 < 0 or nota1 > 10):
                print("Entrada inválida.")
                nota1 = float(input("Entre com a nota: "))
            notaAluno2.append(nota1)
            break
        except ValueError:
            print("Só são aceitos números.")
        
    while True:
        try:
            nota1 = float(input("Entre com a nota: "))
            while(nota1 < 0 or nota1 > 10):
                print("Entrada inválida.")
                nota1 = float(input("Entre com a nota: "))
            notaAluno3.append(nota1)
            break
        except ValueError:
            print("Só são aceitos números.")
        
        
def cadastrar():

    nome = str.capitalize(input("\nEntre com o nome do aluno: "))
    while(nome in nomeAluno):
        print("Nome já cadastrado.")
        nome = str.capitalize(input("Entre com o nome do aluno: "))
    while True:
        try:
            int(nome)
            print("Apenas letras são aceitas.")
            nome = str.capitalize(input("\nEntre com o nome do aluno: "))
        except ValueError:
            break
    
    nomeAluno.append(nome)
    
    matricula = input("Entre com a matrícula: ")
    while(matricula in matriculaAluno):
        print("Matrícula já cadastrada.")
        matricula = input("Entre com a matrícula: ")
    while(matricula.isdigit() == False ):
        print("Apenas números são aceitos.")
        matricula = input("Entre com a matrícula: ")
    while(len(matricula) < 10 or len(matricula) > 10):
        print("A matrícula precisa ter dez números.")
        matricula = input("Entre com a matrícula: ")

    matriculaAluno.append(matricula)
    situacao()
    
    
def listar():
    for i in range(len(notaAluno1)):
        nome1 = nomeAluno[i]
        matricula1 = matriculaAluno[i]
        print("\nNome: %s"%nome1)
        print("Matrícula: %s"%matricula1)
        media1 = (notaAluno1[i] + notaAluno2[i] + notaAluno3[i])/3
        if(media1 >= 7):
            print("Aluno aprovado com média: %.1f"%media1)
        else:
            print("Aluno reprovado com média: %.1f"%media1)
        
menu()

escolha = input("O que deseja fazer? ")
while(escolha != "1" and escolha != "2" and escolha != "3" and escolha != "x"):
        print("\nEntrada inválida.")
        escolha = input("\nO que deseja fazer? ")
while(escolha == "1" or escolha == "2" or escolha == "3" or escolha == "x"):
    while(escolha == "1"):
        cadastrar()
        menu()
        escolha = input("O que deseja fazer? ")
        while(escolha != "1" and escolha != "2" and escolha != "3" and escolha != "x"):
            print("\nEntrada inválida.")
            escolha = input("\nO que deseja fazer? ")

    while(escolha == "2"):
        listar()
        menu()
        escolha = input("O que deseja fazer? ")
        while(escolha != "1" and escolha != "2" and escolha != "3" and escolha != "x"):
            
            print("\nEntrada inválida.")
            escolha = input("\nO que deseja fazer? ")
            
    while(escolha == "3"):
        buscar()
        menu()
        escolha = input("O que deseja fazer? ")
        while(escolha != "1" and escolha != "2" and escolha != "3" and escolha != "x"):
            print("\nEntrada inválida.")
            escolha = input("\nO que deseja fazer? ")
            
    while(escolha == "x"):
        print("Programa finalizado.")
        exit()
