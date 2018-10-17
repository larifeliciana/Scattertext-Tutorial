import scattertext #Ferramenta
import spacy #Utilizado pelo próprio scattertext para fazer o preprocessamento
import pandas #Utilizado para formatar o dataset para criação do corpus do scattertext
import pickle #Usado apenas para carregar o dataset, mas você pode carregar da forma que quiser


#Carregando o dataset
data = open("data1", 'rb')
documentos, classes = pickle.load(data)

#Passando o seu dataset para o formato de DataFrame do pandas, onde uma tabela será criada para gerar o corpus do scattertext, os textos não devem estar preprocessados, pois o scattertext vai usar o spacy para isso
dict = {"texto":documentos, "classes":classes}
data = pandas.DataFrame(dict)

#Gerando o corpus pelo scattertext, a partir disso você terá acesso a diversas informações úteis sobre o seu dataset
nlp = spacy.load('en')
corpus = scattertext.CorpusFromPandas(data,category_col='classes', text_col='texto', nlp=nlp).build()

#Exemplos
print("Número de documentos: " + str(corpus.get_num_docs()))
print("Tamanho de documentos: "+ str(corpus.get_doc_lengths()))
print("Número de termos: "+ str(corpus.get_num_terms()))

print("Palavras que diferem dos corpus comuns: ")
x = corpus.get_scaled_f_scores_vs_background()
print(list(x.index[0:10]))


#Frequência das palavras nas classes
term_freq_df = corpus.get_term_freq_df()
term_freq_df['positivo'] = corpus.get_scaled_f_scores('positivo')
term_freq_df['negativo'] = corpus.get_scaled_f_scores('negativo')

#Ordenando a frequência das palavras para obter as mais frequentes
termosPositivos = term_freq_df.sort_values(by='positivo', ascending=False)
termosNegativos = term_freq_df.sort_values(by='negativo', ascending=False)
print("Palavras mais frequentes entre as classes positivas: ")
print(list(termosPositivos.index[0:10]))

print("Palavras mais frequentes entre as classes negativas: ")
print(list(termosNegativos.index[0:10]))

#Função para a geração do gráfico interativo, o gráfico será gerado num arquivo html que deve ser aberto no navegador e pode demorar um pouco para carregar
html = scattertext.produce_scattertext_explorer(corpus,
          category='positivo', #classe que ficar no eixo y
          category_name='Positivo', #Nomeando a classe apenas para visualização no gráfico
          not_category_name='Negativo', #Nomeando a classe do eixo x
          width_in_pixels=1000) #Tamanho do gráfico
open("graficos.html", 'wb').write(html.encode('utf-8'))