#!/usr/bin/python
# -*- coding: UTF-8 -*-

from scipy.io.arff import loadarff
from scipy.io import arff
from pandas import *
from copy import deepcopy

import Dataset as ds

class Arff(object):
    """ Cria e preenche um arquivo arff com um conjunto de amostras
    
        Dados a classe e os vetores de atributos de um problema, esta classe
        cria um arquivo arff contendo as informacoes passadas por parâmetro
        
        Attributes:
            _a_file: Armazena o arquivo
            _file_name: Nome do arquivo
    """
    _a_file = None
    _file_name = None
    
    
    def __init__(self, file_name = None):
        """Inicializa a classe. Caso um nome de arquivo não seja passado
            um arquivo com nome padrao será criado
            
            Args:
                file_name: Nome do arquivo
        """
        
        if file_name != None:
            
            try:
                self._a_file = open(file_name,'w')
                self._file_name = file_name
            except IOError:
                print "Can't open the file!!"
    
    def __exit__(self):
        """Fecha o arquivo aberto"""
        try:
            self._a_file.close()
        except IOError:
            print "Can't close the %s file!!" % (self._file_name)
    
    def insert_arff_header(self, name = 'foo', attributes_names = None, attributes_values = None):
        """ Cria o cabeçalho de um arquivo arff
        
            Dadas uma lista contendo o nome dos atributos e outra contendo os
            valores dos mesmos (apenas valores nominais possuem valores), e
            definido então o cabeçalho do arquivo
            
            Args:
                name: nome do dataset. E opcional.
                attributes_names: Lista contendo o nome dos atributos
                attributes_values: Lista contendo os valores dos atributos
                    se o atributo for nominal a posição correspondete ao atri-
                    buto contém uma lista dos possíveis valores. Caso a posi-
                    ção do valor do atributo contenha o valor None, então o
                    atributo e do tipo numerico.
                    
            Retuns:
                Nada
        """
        #Nome do dataset
        self._a_file.write("@relation   %s\n\n" % name)
        
        for index in range(len(attributes_names)):
            
            att = None
            
            if attributes_values[index] != None:
                #l e utilizado para otimizar a juncao de strings
                l = []
                l.append("{%s" % attributes_values[index][0])
                
                for i in range(1,len(attributes_values[index])):
                    l.append(",%s" % attributes_values[index][i])
                l.append("}")
                
                nominal = ''.join(l)
                att = "@attribute       %s      %s\n" % (attributes_names[index],nominal)
            else:
                att = "@attribute       %s      numeric\n" % (attributes_names[index])
                
            self._a_file.write(att)
            
    def insert_features(self, attributes_values = None, attributes_classes = None):
        """Insere o vetor de atributos de cada amostra no arquivo arff
        
            Args:
                attributes_values: Uma lista de listas que para cada amostra
                    um vetor de atributos vem associado
                attributes_classes: Uma lista que contém as classes de cada
                    amostra.
            
            Returns:
                Nada
        """
        #Inicia a secao de dados
        self._a_file.write("\n@data\n")
        
        for index in range(len(attributes_values)): #Para cada amostra
            for i in range(len(attributes_values[index])):  #valores 
                self._a_file.write("%s," % attributes_values[index][i])
            
            #Escreve atributo classe
            self._a_file.write("%s\n" % attributes_classes[index])
            
    def load_arff(self, a_file = None):
        """Carrega um arquivo Arff
                
            Args:
                a_file: nome de um arquivo arff.
        """
        
        print 'Loading Arff file %s' % (a_file)
        #data contém os valores de todas as amostras
        #meta contém o cavecalho do arquivo
        data,meta = arff.loadarff(a_file)
        print 'Load complete!!'
        
        #O formato DataFrame e o formato de dados utilizados nos
        #classificadores do ScikitLearn
        data_frame = DataFrame(data)
        
        #Recebe as classes de todas as amostras
        classes = data_frame['class']
        
        #Receve as classes do problema sem repeticoes
        s = np.unique(classes)
        
        #_class_map e um mapa que atribui um indice a cada classe
        #Por exemplo: Acara:0,Dourado:1,...
        class_map = Series([x[0] for x in enumerate(s)], index = s)

        print "Problem Classes:"
        print class_map
        
        #self._classes agora contém as classes de todas as amostras
        #por exemplo:
        # amostra 1 : Dourado
        # amostra 2 : Dourado
        # amostra 3 : Acara
        classes = classes.map(class_map)
        
        #Deleta a coluna das classes das amostras. Assim os dados podem ser
        #utilizados no treinamento.
        #A partir de agora self._data_frame contém todas as amostras com todos
        #os atributos ecxeto o atributo classe que estao no atributo classes
        del data_frame['class']
        
        
        return ds.Dataset(data,meta,data_frame,classes,class_map)
    
    def set_file_name(self, a_name):
        """Define um nome para o arquivo
        
            Args:
                a_name: string contendo o nome do arquivo
        """
        
        if self._a_file != None:
            self._a_file.close()
            
        try:
            self._a_file = open(a_name,'w')
            self._file_name = a_name
        except IOError:
            print "Can't open the file!!"
        
    
    


#def main():
#    a =Arff()
#    
#    features = [[1,2],[3,4],[5,6]]
#    classes = ['b','a','c']
#    
#    
#    #a.insert_features(features,classes)
#    dataset = a.load_arff("fish_Dic64.arff")
#    
#    name = dataset.get_name()
#    names,values = dataset.get_attributes_declarations()
#    features,classes = dataset.get_features()
#    
#    a.insert_arff_header(name,names,values)
#    a.insert_features(features,classes)
#
#
#if __name__ == '__main__':
#    main()