#Mean length of paragraphs
import nltk
import numpy as np
def palavras_tokenizer(texto):
    punct = list("!\"\#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~")
    tokens = nltk.word_tokenize(texto)
    palavras = [i for i in tokens if i not in punct]
    return palavras

def aux_frases_por_paragrafo(texto):
    paragrafos = texto.split("\n")
    frases_por_paragrafo = []
    for i in paragrafos:
        frases_por_paragrafo.append(len(nltk.sent_tokenize(i)))
    return np.array(frases_por_paragrafo)


def aux_palvras_por_sentenca(texto):
    frases = nltk.sent_tokenize(texto)
    palavras_por_sentencas = []
    for i in frases:
        palavras_por_sentencas.append(len(palavras_tokenizer(i)))
    return np.array(palavras_por_sentencas)


def aux_letras_por_palavra(texto):
    palavras = palavras_tokenizer(texto)
    letras_por_palavra = []
    for i in palavras:
        letras_por_palavra.append(len(i))
    return np.array(letras_por_palavra)



def aux_conta_palavra(texto, palavra):
    palavras = palavras_tokenizer(texto)
    return palavras.count(palavra.lower())+palavras.count(palavra.capitalize())



###MÉDIA DO NÚMERO DE SENTENÇAS NOS PARAGRÁFOS DO TEXTO
def DESPL(texto):
    frases_por_paragrafo = aux_frases_por_paragrafo(texto)
    return np.mean(frases_por_paragrafo)



###DESVIO PADRÃO DO NÚMERO DE SENTENÇAS NOS PARAGRÁFOS DO TEXTO
def DESPLd(texto):
    frases_por_paragrafo = aux_frases_por_paragrafo(texto)
    return np.std(frases_por_paragrafo)



###MÉDIA DO NÚMERO DE PALAVRAS NAS SENTENÇAS DO TEXTO.
def DESSL(texto):

    palavras_por_frases = aux_palvras_por_sentenca(texto)
    return np.mean(palavras_por_frases)



###DESVIO PADRÃO DO NÚMERO DE PALAVRAS NAS SENTENÇAS DO TEXTO.
def DESSLd(texto):
    palavras_por_frases = aux_palvras_por_sentenca(texto)
    return np.std(palavras_por_frases)



###MÉDIA DE NÚMERO DE LETRAS POR PALAVRAS NO TEXTO
def DESWLlt(texto):
    letras_por_palavra = aux_letras_por_palavra(texto)
    return np.mean(letras_por_palavra)


###DESVIO PADRÃO DO NÚMERO DE LETRAS POR PALAVRAS NO TEXTO
def DESWLltd(texto):
    letras_por_palavra = aux_letras_por_palavra(texto)
    return np.std(letras_por_palavra)

def WRDPRP1s(texto):
    return aux_conta_palavra(texto, 'eu')

def WRDPRP1p(texto):
    return aux_conta_palavra(texto, 'nós')

def WRDPRP2s(texto):
    return aux_conta_palavra(texto, 'tu')+aux_conta_palavra(texto,'você')

def WRDPRP2p(texto):
    return aux_conta_palavra(texto, 'vós') + aux_conta_palavra(texto, 'vocês')


def WRDPRP3s(texto):
    return aux_conta_palavra(texto, 'ele')+aux_conta_palavra(texto,'ela')

def WRDPRP3p(texto):
    return aux_conta_palavra(texto, 'eles') + aux_conta_palavra(texto, 'elas')

texto = open("texto","r",encoding="utf-8").read()

print(DESSL(texto))