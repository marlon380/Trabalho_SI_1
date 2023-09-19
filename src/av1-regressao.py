import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("aerogerador.dat")

X = data[:,0:1]
N,p = X.shape
y = data[:,1].reshape(N,1)
R = 1000 # Define a quantidade de rodadas
MSE_OLS_C = []
MSE_OLS_S = []
MSE_OLS_K = []
MSE_MEDIA = []
doMirrorGraph = True

if doMirrorGraph:
    # Visualização inicial dos dados através do gráfico de espalhamento.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0],X[:,0],y[:,0],color='orange',edgecolors='k')
    plt.show()

#Função para tratar os dados
def dataTreatment():   
    #Embaralhamento das amostras
    seed = np.random.permutation(N)
    X_random = X[seed,:]
    y_random = y[seed,:]

    #Divide 80% dos dados para treino
    X_treino = X_random[0:int(N*.8),:]  
    y_treino = y_random[0:int(N*.8),:]  

    #Divide 20% dos dados para teste
    X_teste = X_random[int(N*.8):,:]
    y_teste = y_random[int(N*.8):,:]

    return X_treino, y_treino, X_teste, y_teste, X_random, y_random

#Função para calcular a média
def calcAverage(Y):
    b_media = np.mean(Y)
    b_media = np.array([
        [b_media],
        [0],
    ])
    return b_media

#Função para calcular Mínimos quadrados ordinários sem interceptor
def calcOLSWithoutInterceptor(X, Y):
    b_OLS_S = np.linalg.pinv(X.T@X)@X.T@Y
    zero = np.zeros((1,1))
    b_OLS_S = np.concatenate((zero,b_OLS_S),axis=0)

    return b_OLS_S

#Função para calcular Mínimos quadrados ordinários com interceptor
def calcOLSWithInterceptor(X, Y):
    ones = np.ones((X.shape[0],1))
    X_C = np.concatenate((ones,X),axis=1)
    b_OLS_C = np.linalg.pinv(X_C.T@X_C)@X_C.T@Y

    return b_OLS_C

#Função para calcular Mínimos quadrados ordinários regularizado (tikhonov)
def calcOLSRegularizado(X, Y):
    # Adiciona uma coluna de 1s para o termo de intercepto
    ones_k = np.ones((X.shape[0],1))
    X_k = np.concatenate((ones_k, X), axis=1)
    alpha = 0.1

    # Calcula os coeficientes MQO regularizados (Tikhonov)
    I = np.identity(X_k.shape[1])
    b_OLS_K = np.linalg.inv(X_k.T @ X_k + alpha * I) @ X_k.T @ Y

    return b_OLS_K

#Função que realiza a predição
def dataPrediction(X_teste,b_media, b_OLS_C, b_OLS_S, b_OLS_K):
    ones = np.ones((X_teste.shape[0],1))
    X_teste = np.concatenate((ones,X_teste),axis=1)
    
    y_pred_media = X_teste@b_media
    y_pred_ols_c = X_teste@b_OLS_C
    y_pred_ols_s = X_teste@b_OLS_S
    y_pred_ols_k = X_teste@b_OLS_K

    return y_pred_media, y_pred_ols_c, y_pred_ols_s, y_pred_ols_k

#Rodadas de treinamento e testes do model
for r in range(R):

    (X_treino, y_treino, X_teste, y_teste, X_random, y_random) = dataTreatment()
    
    b_media = calcAverage(y_treino)
    b_OLS_C = calcOLSWithInterceptor(X_treino, y_treino)
    b_OLS_S = calcOLSWithoutInterceptor(X_treino, y_treino)
    b_OLS_K = calcOLSRegularizado(X_treino, y_treino)

    (y_pred_media, y_pred_ols_c, y_pred_ols_s, y_pred_ols_k) = dataPrediction(X_teste, b_media, b_OLS_C, b_OLS_S, b_OLS_K)

    MSE_MEDIA.append(np.mean((y_teste-y_pred_media)**2))
    MSE_OLS_C.append(np.mean((y_teste-y_pred_ols_c)**2))
    MSE_OLS_S.append(np.mean((y_teste-y_pred_ols_s)**2))
    MSE_OLS_K.append(np.mean((y_teste-y_pred_ols_k)**2))

boxplot = [MSE_MEDIA,MSE_OLS_C,MSE_OLS_S, MSE_OLS_K]
plt.boxplot(boxplot,labels=['Média','OLS com i','OLS sem i', 'Regulazirado'])
plt.show()

bp = 1