import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

dados = load_breast_cancer()

X = dados.data
y = dados.target

correlacoes = []

for i in range(X.shape[1]):
    correlacoes.append(abs(np.corrcoef(X[:, i], y)[0, 1]))

indices = np.argsort(correlacoes)[-2:]

X = X[:, indices]

media = np.mean(X, axis=0)
desvio = np.std(X, axis=0)

X = (X - media) / desvio

np.random.seed(42)
ordem = np.random.permutation(len(X))

corte = int(len(X) * 0.8)

treino = ordem[:corte]
teste = ordem[corte:]

X_treino = X[treino]
y_treino = y[treino]

X_teste = X[teste]
y_teste = y[teste]

pesos = np.zeros(2)
bias = 0

taxa = 0.1
epocas = 1000

losses = []

for epoca in range(epocas):
    z = X_treino @ pesos + bias
    previsoes = 1 / (1 + np.exp(-z))

    loss = -np.mean(
        y_treino * np.log(previsoes + 1e-9) +
        (1 - y_treino) * np.log(1 - previsoes + 1e-9)
    )

    losses.append(loss)

    erro = previsoes - y_treino

    pesos -= taxa * (X_treino.T @ erro) / len(X_treino)
    bias -= taxa * np.mean(erro)

probabilidades = 1 / (1 + np.exp(-(X_teste @ pesos + bias)))
previsoes = (probabilidades >= 0.5).astype(int)

acuracia = np.mean(previsoes == y_teste)

print("Acuracia:", acuracia)

plt.plot(losses)
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.title("Curva de Aprendizado")
plt.show()

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

x1, x2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

pontos = np.c_[x1.ravel(), x2.ravel()]

z = 1 / (1 + np.exp(-(pontos @ pesos + bias)))
z = z.reshape(x1.shape)

plt.contour(x1, x2, z, levels=[0.5])
plt.scatter(X_teste[:, 0], X_teste[:, 1], c=y_teste)
plt.xlabel(dados.feature_names[indices[0]])
plt.ylabel(dados.feature_names[indices[1]])
plt.title("Fronteira de Decisao")
plt.show()
