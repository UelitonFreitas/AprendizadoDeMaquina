# Visão Geral

O conjunto de códigos aqui apresentado implementa algumas funcionalidades voltadas para a análise de desempenhos e otimizações de classificadores treinados a partir de arquivos ARFF.
Juntamente com a biblioteca [SK-Learn](http://scikit-learn.org/stable/#) implementada em Python, é possiível comaprar e encontrar os melhores
parâmetros de três tipos de classificadores, SVM, árovere de decisão e KNN, todos com implementações fornecidas pelo SK-Learn.
As comparações são baseadas nas métricas: Medida-F (F-Score), AUC(Area Under Curve) e métricas consideradas simples tais como TVP e VP.

Todos os experimentos feitos, quando possível, podem ser executados em paralelo diminuindo consideravelmente o tempo gasto na produção dos resultados. Mas consequentemente,
almentando os recursos utilizados para o processamento tias como memória e núcleos do processador. Uma execução distribuida também pode ser utilizada.
Nos arquivos os passos para tal fim serão explicados com maior detalhes.

# Executando os Scripts
A seguir serão mostradas algumas das operações feitas por este projeto e os arquivos a serem executados para tal fim.
Para a execução de qualquer script basta o utilizad o seguinte passo:

*Dado um arquivo arff de nome Dataset.arff e seja desejada utilizar o script fscore_best_classifiers.py, basta deigitar no terminal:
```
python fscore_best_classifiers.py Dataset.arff
```

## Encontrar os melhores parâmetros dos classificadores utilizando Medida-F (F-Score)
Neste projeto é buscado os melhores parâmetros de três classificadores, Árvore de Decisão, KNN e SVM. O critériooadrão utilziado para avaliar o desempenho de cada classificador
é baseado na Medida-F.

O script fscore_best_classifiers.py, dado um dataset, encontra os melhores parâmetros dos classificadores citados implementando uma Grid Search fornecida pelo [SK-Learn](http://scikit-learn.org/stable/#)
utilizando validação cruzada.

Um exemplo de saída da execução do script fscore_best_classifiers.py pode ser observada a seguir, onde o dataset utilizad é um conjunto de características extraidas de imagens de várias expécies
de peixes:

    Loading Arff file BoC_Dictionary_16.arff
    Load complete!!
    Problem Classes:
    acara-bandeira                 0
    acara-bandeira-marmorizado     1
    acara-disco                    2
    barbus-ouro                    3
    barbus-sumatra                 4
    beta                           5
    carpa                          6
    carpa-media                    7
    dourado                        8
    kinguio                        9
    kinguio-cometa-calico         10
    kinguio-korraco               11
    mato-grosso                   12
    molinesia-preta               13
    oscar                         14
    oscar-albino                  15
    pacu                          16
    palhaco                       17
    papagaio                      18
    paulistinha                   19
    piau-tres-pintas              20
    platy-laranja                 21
    platy-rubi                    22
    platy-sangue                  23
    telescopio                    24
    tetra-negro                   25
    tricogaster                   26
    tucunare                      27
    dtype: int64

    ----------------------------------------------------
    Finding best C for SVC.

    Running RandomizedSearchCV....
    Done!!

    Best estimator found:
    SVC(C=18294.966571724693, cache_size=1000, class_weight=None, coef0=0.0,
    degree=3, gamma=0.0004048653310876237, kernel='rbf', max_iter=-1,
    probability=False, random_state=None, shrinking=True, tol=0.001,
    verbose=False)

    Running cross validation...
    Running fold 0...
    Running fold 1...
    Running fold 2...
    Running fold 3...
    Running fold 4...
    Running fold 5...
    Running fold 6...
    Running fold 7...
    Running fold 8...
    Running fold 9...
    Report:
                                precision     recal f-measure   support
    
            acara-bandeira          0.68      0.72      0.68         4
     acara-band-armorizado          0.41      0.53      0.44         4
               acara-disco          0.62      0.70      0.63         4
               barbus-ouro          0.57      0.53      0.51         4
            barbus-sumatra          0.84      0.82      0.83         4
                      beta          0.73      0.75      0.67         4
                     carpa          0.72      0.90      0.79         4
               carpa-media          0.62      0.53      0.55         4
                   dourado          0.53      0.57      0.54         4
                   kinguio          0.70      0.57      0.60         4
     kinguio-cometa-calico          0.83      0.93      0.87         4
           kinguio-korraco          0.67      0.65      0.65         4
               mato-grosso          0.84      0.82      0.82         4
           molinesia-preta          0.79      0.60      0.67         4
                     oscar          0.75      0.72      0.72         4
              oscar-albino          0.73      0.80      0.75         4
                      pacu          0.64      0.40      0.45         4
                   palhaco          0.44      0.40      0.41         4
                  papagaio          0.53      0.45      0.47         4
               paulistinha          0.57      0.60      0.54         4
          piau-tres-pintas          0.42      0.45      0.41         4
             platy-laranja          0.50      0.50      0.48         4
                platy-rubi          0.58      0.50      0.51         4
              platy-sangue          0.78      0.42      0.53         4
                telescopio          0.80      0.82      0.77         4
               tetra-negro          0.52      0.57      0.53         4
               tricogaster          0.87      0.80      0.81         4
                  tucunare          0.60      0.62      0.59         4

               avg / total       0.65      0.63      0.62       4.0

                             0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27

            acara-bandeira  29  3  0  0  0  0  0  1  0  0  0  0  0  1  0  0  0  0  0  2  0  1  1  0  2  0  0  0
     acara-band-armorizado   1 21  1  1  0  1  0  0  3  0  0  0  1  1  0  0  0  2  0  1  0  1  3  2  1  0  0  0
               acara-disco   0  0 28  0  0  0  0  1  0  0  0  1  0  0  0  0  1  1  0  1  0  3  0  0  0  3  0  1
               barbus-ouro   0  0  0 21  1  4  3  0  1  0  4  0  1  0  1  0  0  3  0  0  1  0  0  0  0  0  0  0
            barbus-sumatra   0  0  0  2 33  0  0  0  0  0  0  0  0  0  0  0  0  2  1  0  0  0  0  0  0  0  0  2
                      beta   2  0  0  2  0 30  0  0  0  0  0  0  0  0  1  0  3  0  1  0  0  0  1  0  0  0  0  0
                     carpa   0  0  0  0  1  0 36  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  1  0
               carpa-media   1  0  2  0  0  0  0 21  0  1  0  1  0  0  0  0  0  0  1  2  2  2  0  0  1  6  0  0
                   dourado   0  4  0  1  0  0  0  0 23  0  0  0  0  0  0  1  0  4  0  0  6  0  0  0  0  0  0  1
                   kinguio   0  2  1  0  0  2  0  0  0 23  0  3  0  0  0  0  0  0  3  1  0  1  2  0  1  1  0  0
     kinguio-cometa-calico   0  0  0  2  0  0  0  0  0  0 37  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           kinguio-korraco   2  0  3  0  1  0  0  3  0  1  0 26  0  0  0  0  0  0  0  0  0  0  2  1  0  1  0  0
               mato-grosso   0  2  0  3  0  0  0  0  0  0  1  0 33  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
           molinesia-preta   1  4  1  1  0  1  1  1  0  1  0  0  1 24  0  0  2  0  0  1  0  0  0  1  0  0  0  0
                     oscar   0  0  0  0  0  3  0  0  0  0  0  0  1  0 29  0  0  0  7  0  0  0  0  0  0  0  0  0
              oscar-albino   0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 32  3  0  0  0  3  0  0  0  0  0  0  1
                      pacu   0  2  0  0  0  2  3  0  0  0  0  0  1  1  0  9 16  0  0  0  0  0  1  0  1  1  2  1
                   palhaco   0  1  1  2  2  0  2  0  8  0  1  0  0  0  0  0  0 16  1  2  1  2  0  0  1  0  0  0
                  papagaio   2  2  0  0  0  0  0  1  1  1  0  0  2  0  4  0  0  1 18  0  1  3  1  0  0  1  0  2
               paulistinha   0  0  2  0  0  0  0  0  1  1  0  3  0  2  0  1  0  2  0 24  1  0  0  0  1  0  0  2
          piau-tres-pintas   1  2  0  0  0  0  3  0  5  0  0  0  0  0  0  1  0  1  1  2 18  0  0  0  0  0  3  3
             platy-laranja   1  0  4  0  0  0  0  0  0  0  0  3  0  0  0  0  0  1  2  3  0 20  0  1  1  3  0  1
                platy-rubi   5  6  0  0  0  1  0  0  0  3  0  1  0  0  2  0  0  0  0  0  0  0 20  1  0  1  0  0
              platy-sangue   0  2  2  2  0  1  0  1  0  4  1  0  1  0  0  0  0  0  1  1  0  1  6 17  0  0  0  0
                telescopio   1  0  0  1  0  0  0  0  0  0  0  1  0  1  0  0  0  0  0  3  0  0  0  0 33  0  0  0
               tetra-negro   0  0  1  0  0  0  0  7  0  2  0  0  0  0  0  0  0  1  1  1  1  3  0  0  0 23  0  0
               tricogaster   0  0  0  1  0  0  1  0  0  0  1  0  0  0  0  0  4  0  0  0  1  0  0  0  0  0 32  0
                  tucunare   1  0  0  0  2  0  1  0  0  0  0  0  0  0  0  2  0  0  1  1  5  0  0  0  2  0  0 25


Os resultados mostrados anteriormente mostram os melhores parâmetros encontrados para o o classificador SVM e também
foi fornecida a matriz de confusão do experimento. Para encontrar os melhores parâmetros foi utilziada a validação
cruzada contendo 10 dobras e 10 repetições. Para gerar a matriz de confusão a validação cruzada foi novamente aplicada
utilizando o classificador otimizado.

Ao término da execução também é gerado um pdf contendo um gráfico da matriz de confusão gerada para cada experimento.

## Gerar gráfico da AUC com Medida-F
O script  svm_auc_report.py gera um gráfico roc de um classificador SVM. O gráfico é gerado utilizando a técnica 1 vs Todos (1xall) com validação cruzada de 10 partições e repetições. O gráfico mostrado contém a curva e AUC de cada classe e a média das mesmas.

# Instalação

## Antes de executar os programas é necessáiro instalar algumas dependências:

### Linux - Instale as dependências do scikit-learn (basta colar no terminal):

``` 
sudo apt-get install build-essential python-dev python-setuptools \
                     python-numpy python-scipy \
                     libatlas-dev libatlas3gf-basesudo update-alternatives --set libblas.so.3 \
```


``` 
sudo update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3 
```

```
sudo apt-get install python-matplotlib
```

```
sudo apt-get install python-sklearn 
```

```
sudo apt-get install python-pip
```

```
sudo pip install pip
```

### Windows - No Windows instale o pacote [Anaconda](http://continuum.io/downloads).

-----------------------------------------------------------------------------------------------

