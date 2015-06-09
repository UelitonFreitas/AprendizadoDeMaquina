#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class Dataset(object):
    """Esta calsse representa um dataset carregado a partir de um arquivo arff
        
        Atributes:
            _name: nome do dataset
            _data: contém os dados das amostras vindos do arff.
            data: contém os mesmos dados de _data mas no formato array do numpy
            _meta: contém o cabeçalho do arquivo arff.
            _dataFrame: contém os dados do aquivo no formato DataFrame
            _utilizados nos classificadores
            _classes: contém a lista de classes encontradas no arff
            target: ontem os mesmos dados de _classes mas no formato array do np
            _mapping: contém o mapa das classes
            _scaled_data: contém os valores das amostras normalizados
            _class_len: Quantidade de classes no dataset
    """
    _name = None
    data = None
    target = None
    _data = None
    _header = None
    _data_frame = None
    _features_classes = None
    _classes_map = None
    _class_len = None
    
    def __init__(self,data,header,data_frame,features_classes,class_map):
        """Construtor"""
        self._name = header.name
        self._data = data
        self._header = header
        self._data_frame = data_frame
        self._features_classes = features_classes
        self._classes_map = class_map
        self._class_len = len(class_map)
        
        self.data = np.array(data_frame)
        self.target = np.array(features_classes)
        
        
    def __exit__(self):
        """Destrutor"""
        del self._data
        del self._header
        del self._features_classes
        del self._classes_map
    
    def get_features(self):
        """Retorna as listas de atributos das amostras e o vetor contendo as
            classes
        
            Retorna uma lista de listas que cada posição contém uma lista
            contendo os valores dos atributos das amostras. A segunda lista
            retornada contém as classes das amostras, portanto, cada posição
            das listas correspondem a uma amostra.
        
            Returns:
                features: uma lista em que cada posição representa uma amostra.
                    Cada posição e formada por uma outra lista que contém os
                    valores dos atributos.
                classes: Uma lista em que cada posição i contém o atributo
                    classe da amostra i
        """
        
        features = []
        classes = []
        for i in range(len(self._data)):
            f = []
            for j in range(0,len(self._data[i])-1):
                f.append(self._data[i][j])
            features.append(f)
            classes.append(self._data[i][-1])
        
        return features,classes
    
    def turn_index_to_names(self, class_index = None):
        """Recebe uma lista contém os índices das classes do dataset
            e retorna uma lista contendo os nomes das classes ao invés
            dos índices
           
            Args:
                class_index: lista de índices das classes
            Return:
                class_name: lista de nomes das classes a partir dos índices
        """
        
        class_names = []
        for index in range(len(class_index)):
            class_names.append(self.get_class_name_by_index(class_index[index]))
        
        return class_names
    
    
    def get_attributes_declarations(self):
        """Retorna duas listas. Uma contém os nomes dos atributos e a outra
            contém os valores dos atributos.
            
            A lista que contém os valores dos atributos define que tipo de
            atributo o  mesmo é. Caso o atributo seja numérico, nenhum valor
            será passado, caso contrário, a posição correspondente a amostra
            contém os valores possíveis dos atributos nominais.
            
            Returns:
                attributes_names: Lista contendo os nomes dos atributos.
                attributes_values: contém os valores possíveis dos atributos
                    Dado um atributo, há dois valores possíveis:
                        None - o atributo e do tipo número
                        {'a','b','c'}  - valores possíveis para o atributo
                        nominal.
                        
                Por exemplo:
                    attributes_names = ['a1','a2']
                    attributes_values = [None,{'a','b','c'}]
                    
                    representam os atributos:
                        a1 numeric
                        a2 {a,b,c} nominal
        
        """
        
        attributes_names = []
        attributes_values = []
        for name in self._header:
            attributes_names.append(name)
            value = None
            if self._header[name][0] == 'nominal':
                value = self._header[name][1]
            attributes_values.append(value)
        
        return attributes_names,attributes_values
    
    def get_class_names(self, ):
        """Retorna uma lista contendo os nomes das classes do dataset
        
            Retiurns:
                Lista contendo o nome de cada classe.
        """
        names = []
        for c in  self._classes_map.iteritems():
            names.append(c[0])
        
        return names


    def get_class_name_by_index(self,index):
        """Retorna o nome da classe no index
            
            Retorna o nome da classe no índice index
            
            Args:
                index: índice da class no mapa de classes
                
            Returns:
                O nome da classe ou caso contrário None
        """
        
        #Esta condicao pode ser ignorada caso a utilziacao de dataset 1xall
        #não seja utilizada.
        #Como cada classe possui um indece, e para algumas operacoes futuras
        #e utilizado uma classe qualquer (para dataset 1xall) esta classe
        #qualquer e denominada foo e possui o índice como sendo o número de
        #classes + 1
        if index == len(self._classes_map):
            return "foo"
        
        for c in  self._classes_map.iteritems():
            if c[1] == index:
                return c[0]
        return None
    
    def get_name(self):
        """Retorna o nome do dataset"""
        return self._name
    
    
    def get_classes(self):
        """Retorna as classes das amostras
        
            Returns:
                _features_classes
        """
        return self._features_classes
    
    
    def get_class_map(self):
        """Retorna o mapa das classes
        
            Returns:
                _classes_map
        """
        
        return self._classes_map
    
    def get_class_len(self):
        """Retorna o número de classes do dataset
        
        Returns:
            O número de classes contido no dataset
        """
        return self._class_len
    
    def get_class_index_by_name(self, name):
        """Retorna o indoce da classe como nome name
            
            Args:
                name: nome da classe
                
            Returns:
                índice da classe
            
            Raises:
                IndexError: A classe não foi encontrada
        """
        try:
            
            if name == "foo":
                return len(self._classes_map)
            
            return self._classes_map[name]
        except IndexError:
            print "%s is not a valid key" % name
    