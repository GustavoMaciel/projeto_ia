import word_suggestor as lib
import re
from datetime import datetime


option = -1
while option != 0:
    print ("Choose one option: \n{1} Bigram\n{2} Trigram\n{0} Exit")
    try:
        option = int(input("> "))
        if option not in [0, 1, 2]:
            raise Exception
    except:
        print("Invalid option.")
        
    if option in [1,2]:
        algorithm = ""
        if option == 1:
            algorithm = "bigram"
        elif option == 2:
            algorithm = "trigram"
            
        phrase = input("Type a sentence:\n> ")
        phrase_tokens = re.findall(lib.regex, phrase.lower())

        if(option == 2 and len(phrase_tokens) < 2):
            print("Two or more words are necessary for the trigram option.")
            continue
        elif len(phrase_tokens) < 1:
            print("At least one word is necessary for this option to work.")
            continue

        print("Calculating results...")
        starting_time = datetime.now().time()
        result = lib.shannon(phrase_tokens, algorithm, 3)
        finish_time = datetime.now().time()

        print("Our suggestions are: ")        
        print(phrase + " " + result[0])     
        print(phrase + " " + result[1])     
        print(phrase + " " + result[2])
        
        print("Start: {}\nFinish: {}\n\n".format(starting_time, finish_time))

        
        
