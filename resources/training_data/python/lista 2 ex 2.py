marcilio = 0
aurelio = 0
branco = 0

for cont in range(4):
    voto = str.lower(input("Entre com seu voto: "))
    
    if(voto == "marcílio" or voto == "marcilio"):
        marcilio += 1
        
    elif(voto == "aurélio" or voto == "aurelio"):
        aurelio += 1
        
    elif(voto == "branco"):
        branco += 1
        
    else:
        print("Entrada inválida.")

if(marcilio > aurelio and marcilio > branco):
    print("Marcílio é o vencedor.")

elif(aurelio > marcilio and aurelio > branco):
    print("Aurélio e o vencedor.")

else:
    print("Uma nova votação é necessária.")
