FilmesNetflix = [ ["Os Miseráveis", "Musical", 2012, 158, "Português"], ["Gravidade", "Aventura", 2013, 91, "Inglês"], ["Frozen", "Infantil", 2013, 102, "Português"], ["O Artista", "Comédia", 2011, 100, "Inglês"], ["Os Smurfs", "Infantil", 2011, 86, "Português"]]
nome = 0
tipo = 1
ano = 2
duracao = 3
idioma = 4

FilmesNetflix[1][duracao]  = 95 #a)

for i in range(len(FilmesNetflix)): #b)
    if(FilmesNetflix[i][ano] == 2011):
        print(FilmesNetflix[i][nome], FilmesNetflix[i][duracao], FilmesNetflix[i][idioma])


maisCurto = 10000   #c)
for i in range(len(FilmesNetflix)):     
    if(FilmesNetflix[i][idioma] == "Inglês"):
        if(FilmesNetflix[i][duracao] < maisCurto):
            maisCurto = FilmesNetflix[i][duracao]
            filmeMaisCurto = FilmesNetflix[i][nome]
print("O filme mais curto:",filmeMaisCurto)


total = 0 #d)
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][tipo] == "Infantil"):
        total += FilmesNetflix[i][duracao]
print(total)

#e)
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][idioma] == "Inglês"):
        print(FilmesNetflix[i][nome], FilmesNetflix[i][ano])

#f)
qtdMenor100 = 0
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][duracao] < 100):
        qtdMenor100 += 1
print("Quantidade de filmes menor que 100 minutos:",qtdMenor100)

#g)
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][idioma] == "Português" and FilmesNetflix[i][ano] == 2011):
        print("Português e lançado em 2011:" ,FilmesNetflix[i][nome])


#h)
somatoriaInfantis = 0
qtdInfantil = 0
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][tipo] == "Infantil"):
        qtdInfantil += 1
        somatoriaInfantis += FilmesNetflix[i][duracao]
resultadoEsperado = somatoriaInfantis//qtdInfantil
print("Media filmes infantis:", resultadoEsperado)


#i)
listaDeLancados = []
for i in range(len(FilmesNetflix)):
    if(FilmesNetflix[i][ano] >= 2012 and FilmesNetflix[i][ano] <= 2015):
        listaDeLancados.append(FilmesNetflix[i][tipo])

print(listaDeLancados)


