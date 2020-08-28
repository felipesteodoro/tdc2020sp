
import random
import numpy as np
#pip install deap
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import multiprocessing

dataset = pd.read_csv("yourdataset.csv")
y = dataset["class"].values
dataset.drop(["class"], inplace = True,  axis = 1) 

X = dataset.values
scalar = StandardScaler().fit(X)
X = scalar.transform(X)

classifier = SVC(C = 100, gamma = 0.0001, kernel = 'rbf')

#Adaptado de https://github.com/kaushalshetty/FeatureSelectionGA
def calculate_fitness(individual):
          np_ind = np.asarray(individual)
          if np.sum(np_ind) == 0:
              return  (0.0,)
          else:
              feature_idx = np.where(np_ind==1)[0]
              x_temp = X[:,feature_idx]  
          cv_set = np.repeat(-1.,x_temp.shape[0])
          skf = StratifiedKFold(n_splits = 5)
          for train_index,test_index in skf.split(x_temp,y):
              X_train,X_test = x_temp[train_index],x_temp[test_index]
              y_train,y_test = y[train_index],y[test_index]
              if X_train.shape[0] != y_train.shape[0]:
                  raise Exception()
              classifier.fit(X_train,y_train)
              predicted_y = classifier.predict(X_test)
              cv_set[test_index] = predicted_y
          
          acc = accuracy_score(y, cv_set)
          
          return (acc,)


toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", calculate_fitness)
toolbox.register("mate", tools.cxUniform,  indpb=0.3)
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
toolbox.register("select", tools.selBest)



if __name__ == "__main__": 
    
    random.seed(25)
    MU, LAMBDA = 200, 400
    populacao = toolbox.population(n = MU)
    probabilidade_crossover = 0.8
    probabilidade_mutacao = 0.2
    numero_geracoes = 200
    
    pool = multiprocessing.Pool()  
    toolbox.register("map", pool.map)
    
    estatisticas = tools.Statistics(key=lambda individuo: individuo.fitness.values)
    estatisticas.register("max", np.max)
    estatisticas.register("min", np.min)
    estatisticas.register("med", np.mean)
    estatisticas.register("std", np.std)    
    populacao, info = algorithms.eaMuPlusLambda(populacao, toolbox,MU,LAMBDA, probabilidade_crossover, probabilidade_mutacao, numero_geracoes, estatisticas, verbose  = True)
        
    melhores = tools.selBest(populacao, 1)
    
    valores_grafico = info.select("max")
    plt.figure("Evolução")
    plt.plot(valores_grafico)
    plt.title("Acompanhamento dos valores")
    plt.show()
        
    feat_selected = pd.DataFrame(list(melhores[0]), columns  = ["Selected"])
    feat_selected = feat_selected["Selected"] == 1
    dtselected  = dataset[dataset.columns[feat_selected]]
    
    X = dtselected.values
    
    dtselected["class"] = y
    dtselected.to_csv('features_selected_dataset.csv')        
    
    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=25)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    plt.figure("Matriz de Confusão")
    plot_confusion_matrix(classifier, X_test, y_test, normalize = 'true')
    print(classification_report(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))
    
    pool.close()
                













