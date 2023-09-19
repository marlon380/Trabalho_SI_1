import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Data = np.loadtxt("EMG.csv",delimiter=',')

#Visualização inicial dos dados através do gráfico de espalhamento.
colors = ['red','green','purple','blue','gray']
k = 0
for i in range(10):
    for j in range(5):
        plt.scatter(Data[k:k+1000,0],Data[k:k+1000,1],color=colors[j],
                    edgecolors='k')
        k+=1000
plt.show()

#Definição inicial das variaveis
N,p = Data.shape

neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 

ACC_OLS = []
ACC_OLS_K = []
ACC_KNN = []
R = 100

#Função para tratar os dados
def dataTreatment():
    Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))

    s = np.random.permutation(N) # Gera dados aleatorios

    X = Data[s,:]
    Y = Y[s,:]

    X = np.concatenate((
        np.ones((N,1)),X
    ),axis=1)

    X_train = X[0:int(N*.8),:]
    Y_train = Y[0:int(N*.8),:]

    X_test = X[int(N*.8):,:]
    Y_test = Y[int(N*.8):,:]

    return X_train, Y_train, X_test, Y_test

def calcOLSWithInterceptor(X, Y):
    ones = np.ones((X.shape[0],1))
    X_C = np.concatenate((ones,X),axis=1)
    b_OLS_C = np.linalg.pinv(X_C.T@X_C)@X_C.T@Y

    return b_OLS_C

def calcOLSRegularizado(X, Y, i):
    # Adiciona uma coluna de uns para o termo de intercepto
    ones_k = np.ones((X.shape[0],1))
    X_k = np.concatenate((ones_k, X), axis=1)
    alpha = (i + 1) / R

    # Calcula os coeficientes MQO regularizados (Tikhonov)
    I = np.identity(X_k.shape[1])
    b_OLS_K = np.linalg.inv(X_k.T @ X_k + alpha * I) @ X_k.T @ Y

    return b_OLS_K

def calcKNN(X_train, y_train, X_test, k):
    ones_knn = np.ones((X_train.shape[0],1))
    X_knn = np.concatenate((ones_knn, X_train), axis=1)
    indices_amostra = np.random.choice(X_knn.shape[0], 10000, replace=False)
    X_train_novo = X_knn[indices_amostra]
    distances = np.linalg.norm(X_train_novo - X_test, axis=0)

    nearest_neighbors_indices = np.argsort(distances, axis=0)[:k]
    
    nearest_neighbors_labels = y_train[nearest_neighbors_indices]
    
    # Calcule a classe mais frequente entre os vizinhos mais próximos
    y_pred_knn = np.argmax(nearest_neighbors_labels, axis=0)
    
    return y_pred_knn

def modelAccurracy(Y_test, Y_hat, model: str):
    acertos = 0
    if (model == 'KNN'):
        acertos = np.sum(Y_test == Y_hat)
        acuracia = acertos / len(Y_test)
        return acuracia
    else:
        discriminante = np.argmax(Y_hat,axis=1)
        discriminante2 = np.argmax(Y_test,axis=1)

        acertos = discriminante==discriminante2
        acuracia = np.count_nonzero(acertos)/len(Y_test)
        return acuracia

for r in range(R):
    (X_train, Y_train, X_test, Y_test) = dataTreatment()

    (b_OLS_C) = calcOLSWithInterceptor(X_train, Y_train )
    (b_OLS_K) = calcOLSRegularizado(X_train, Y_train, r )

    ones = np.ones((X_test.shape[0],1))
    X_test = np.concatenate((ones,X_test),axis=1)

    y_pred_ols = X_test @ b_OLS_C
    y_pred_ols_k = X_test @ b_OLS_K
    (y_pred_knn) = calcKNN(X_train,Y_train, X_test, 7)

    ACC_OLS.append(modelAccurracy(Y_test, y_pred_ols, model='OLS'))
    ACC_OLS_K.append(modelAccurracy(Y_test, y_pred_ols_k, model='OLS'))
    ACC_KNN.append(modelAccurracy(Y_test, y_pred_knn, model='KNN'))

def statisticAnalysis(vector):
    media = np.mean(vector) # calcula a média
    desvio_padrao = np.std(vector) # calcula o desvio padrão

    # Encontra os valores únicos e suas contagens
    values, counts = np.unique(vector, return_counts=True)
    # Encontra a posição do valor com a contagem máxima
    position = np.argmax(counts)
    moda = values[position] # calcula a moda

    # Calcula o valor máximo
    max = np.max(vector)

    # Calcula o valor mínimo
    min = np.min(vector)

    return media, desvio_padrao, moda, max, min

def OLS_Table():
    (media, desvio_padrao,moda,max,min) = statisticAnalysis(ACC_OLS)

    df = pd.DataFrame({
        'OLS':['1.', '2.', '3.','4.', '5.'],
        'Estatística': ['Média', 'Desvio Padrão', 'Moda', 'Máximo', 'Mínimo'],
        'Valor': [media, desvio_padrao, moda, max, min]
    })

    return print(df.to_string(index=False))

def OLS_K_Table():
    (media, desvio_padrao,moda,max,min) = statisticAnalysis(ACC_OLS_K)

    df = pd.DataFrame({
        'OLS Reg.':['1.', '2.', '3.','4.', '5.'],
        'Estatística': ['Média', 'Desvio Padrão', 'Moda', 'Máximo', 'Mínimo'],
        'Valor': [media, desvio_padrao, moda, max, min]
    })

    return print(df.to_string(index=False))

def KNN_Table():
    (media, desvio_padrao,moda,max,min) = statisticAnalysis(ACC_KNN)

    df = pd.DataFrame({
        'KNN':['1.', '2.', '3.','4.', '5.'],
        'Estatística': ['Média', 'Desvio Padrão', 'Moda', 'Máximo', 'Mínimo'],
        'Valor': [media, desvio_padrao, moda, max, min]
    })

    return print(df.to_string(index=False))

OLS_Table()
OLS_K_Table()
KNN_Table()
bp=1
