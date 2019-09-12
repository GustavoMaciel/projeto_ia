from models import State
from a_star import *

joao_pessoa = State("João Pessoa", 460)
campina_grande = State("Campina Grande", 300)
itabaiana = State("Itabaiana", 360)
santa_rita = State("Santa Rita", 451)
mamanguape = State("Mamanguape", 380)
guarabira = State("Guarabira", 340)
areia = State("Areia", 316)
picui = State("Picui", 250)
soledade = State("Soledade", 243)
coxixola = State("Coxixola", 232)
patos = State("Patos", 122)
monteiro = State("Monteiro", 195)
catole = State("Catolé do Rocha", 110)
pombal = State("Pombal", 55)
itaporanga = State("Itaporanga", 65)
sousa = State("Sousa", 20)
cajazeiras = State("Cajazeiras", 0)

joao_pessoa.add_neighbours([[campina_grande, 125], [itabaiana, 68], [santa_rita, 26]])
campina_grande.add_neighbours([[joao_pessoa, 125], [itabaiana, 65], [areia, 40], [coxixola, 128], [soledade, 58]])
itabaiana.add_neighbours([[joao_pessoa, 68], [campina_grande, 65]])
santa_rita.add_neighbours([[joao_pessoa, 26], [mamanguape, 38]])
mamanguape.add_neighbours([[santa_rita, 38], [guarabira, 42]])
guarabira.add_neighbours([[mamanguape, 42], [areia, 41]])
areia.add_neighbours([[guarabira, 41], [campina_grande, 40]])
picui.add_neighbours([[soledade, 69]])
soledade.add_neighbours([[campina_grande, 58], [patos, 117], [picui, 69]])
coxixola.add_neighbours([[campina_grande, 128], [monteiro, 83]])
patos.add_neighbours([[soledade, 117], [pombal, 71], [itaporanga, 108]])
monteiro.add_neighbours([[coxixola, 83], [itaporanga, 224]])
catole.add_neighbours([[pombal, 57]])
pombal.add_neighbours([[catole, 57], [patos, 71], [sousa, 56]])
itaporanga.add_neighbours([[patos, 108], [monteiro, 224]])
sousa.add_neighbours([[pombal, 56], [cajazeiras, 43]])
cajazeiras.add_neighbours([[sousa, 43], [itaporanga, 121]])

state = search(joao_pessoa, cajazeiras)

path = []
while state is not None:
    path.append(state['state'])
    state = state['parent']

path.reverse()
print(path)
