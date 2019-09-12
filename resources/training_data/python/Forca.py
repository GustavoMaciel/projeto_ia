    ## Definindo as variáveis iniciais do programa
end = "y"
while(end != "n"):
    import time
    import random

    pal = ["PALAVRA","CADEIA","CASA", "PROGRAMA","CORDA","PARTO","NOME","BRASIL","PORTUGAL","DESGRAÇA","PINGUIN","DEFINIR","CACHORRO"]

    x = (len(pal))-1

    sort = random.randint(0,x)

    palavras = list(pal[sort])

    letras = []
    tentativas = [5]
    car = "₪"

    ## Todas as funções necessárias.

    def menu():
        print("\nSeja Bem-Vindo ao Jogo da Forca! \n")
        print("[1] Iniciar\n[2] Regras\n[x] Sair\n")
    def menuP2():
        print("\nO que deseja fazer agora? \n")
        print("[1] Tentar Letra\n[2] Chutar Palavra\n[x] Sair\n")

    def escondido():
        print("A palavra é:")
        for i in range(len(palavras)):
            letras.append(car)
            print (end = letras[i])

        print("\n")
        
    def revelado():
        print("\n")

        if(letra in palavras):
            for i in range(len(palavras)):
                if(letra == palavras[i]):
                    letras[i] = palavras[i]
            print("Você acertou e ganhou uma tentativa.\n")
            tentativas[0] += 1

        elif(letra not in palavras):
            tentativas[0] -= 1
            print("Você errou e perdeu uma tentativa.\n")

        if(tentativas[0] == 0):
            print("Você esgotou o número de tentativas. Mais sorte na próxima.")
            time.sleep(2.2)
            exit()
           
        for i in range(len(palavras)):
            print(end = letras[i])
        print("\n\n")
        
    ## Esta parte é para o primeiro menu, nada de importante acontece aqui.
        
    cond = 0
    while(cond != 1):
        menu()
        
        escolha = str.lower(input("O que deseja fazer? "))
        while(escolha != "1" and escolha != "2"  and escolha != "x"):
                print("Entrada inválida.")
                escolha = input("O que deseja fazer? ")

        if(escolha == "1"):
            print("\n")
            escondido()
            cond = 1
            
        elif(escolha == "2"):
            print("\nOlá, seja bem-vindo!\nNo jogo da forca você irá chutar letras na tentativa de acertar a palavra.\nVocê tem 5 tentativas, para cada erro você perde uma.\nPara cada acerto você ganhara uma tentativa.\nSe o número de tentativas chegar a zero, você perderá.")

        elif(escolha == "x"):
            print("Programa finalizado.")
            time.sleep(2.2)
            exit()

    ## Esta parte é o jogo em si, Tudo de importante ta acontecendo aqui.
            
    while(tentativas[0] > 0 and letras != palavras):
        menuP2()

        escolha = str.lower(input("O que deseja fazer? "))
        while(escolha != "1" and escolha != "2"  and escolha != "x"):
                print("Entrada inválida.")
                escolha = input("O que deseja fazer? ")

        if(escolha == "1"):
            print("\n")
            letra = str.upper(input("Entre com uma letra: "))
            while True:
                    try:
                        int(letra)
                        print("Apenas letras são aceitas.")
                        letra = str.upper(input("Entre com uma letra: "))
                    except ValueError:
                        break
                    
            revelado()
            
        elif(escolha == "2"):
            palavra = str.upper(input("Entre com a palavra: "))

            while True:
                    try:
                        int(palavra)
                        print("Apenas palavras são aceitas.")
                        letra = str.upper(input("Entre com uma palvra: "))
                    except ValueError:
                        break
            
            if(list(palavra) == palavras):
                print("\nParabéns! Você venceu!")
                print("A palavra era: %s"%palavras)
                break
            
            else:
                print("\nVocê errou, a palavra era: %s" %palavras)
                time.sleep(2.2)
                exit()
                
        elif(escolha == "x"):
            print("Programa finalizado.")
            time.sleep(2.2)
            exit()

    ## Por termos atingido o número máximo de tentativas ou termos achado a palavra.

    if(letras == palavras):
        print("Parabéns, você venceu!")
        time.sleep(2.2)

    if(tentativas[0] == 0):
        print("Que pena, você atingiu o máximo de tentativas. Mais sorte na próxima!")
        time.sleep(2.2)

    end = str.lower(input("Deseja jogar novamente? y/n "))
    while(end != "y" and end != "n"):
        print("Esta não é uma entrada válida.")
        end = str.lower(input("Deseja jogar novamente? y/n "))
