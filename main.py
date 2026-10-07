import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# carregando o dataset
dados = load_breast_cancer()

X = dados.data
y = dados.target

print("Formato dos dados:", X.shape)
print("Classes:", dados.target_names)


# escolhendo as duas caracteristicas com maior correlacao com a classe
correlacoes = []

for i in range(X.shape[1]):
    correlacao = np.corrcoef(X[:, i], y)[0, 1]
    correlacoes.append(abs(correlacao))

indices = np.argsort(correlacoes)[-2:][::-1]

X = X[:, indices]

print("\nCaracteristicas escolhidas:")
print(dados.feature_names[indices[0]])
print(dados.feature_names[indices[1]])


# separando treino e teste
np.random.seed(42)

ordem = np.random.permutation(len(X))
corte = int(len(X) * 0.8)

indices_treino = ordem[:corte]
indices_teste = ordem[corte:]

X_treino = X[indices_treino]
y_treino = y[indices_treino]

X_teste = X[indices_teste]
y_teste = y[indices_teste]


# padronizando os dados
media = np.mean(X_treino, axis=0)
desvio = np.std(X_treino, axis=0)

X_treino = (X_treino - media) / desvio
X_teste = (X_teste - media) / desvio


# funcao sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# iniciando os parametros do neuronio
pesos = np.zeros(X_treino.shape[1])
bias = 0

taxa_aprendizado = 0.1
epocas = 1000

historico_loss = []


# treinamento
for epoca in range(epocas):
    z = X_treino @ pesos + bias

    previsoes = sigmoid(z)

    epsilon = 0.000000001

    loss = -np.mean(
        y_treino * np.log(previsoes + epsilon)
        + (1 - y_treino) * np.log(1 - previsoes + epsilon)
    )

    historico_loss.append(loss)

    erro = previsoes - y_treino

    gradiente_pesos = (
        X_treino.T @ erro
    ) / len(y_treino)

    gradiente_bias = np.mean(erro)

    pesos = pesos - taxa_aprendizado * gradiente_pesos
    bias = bias - taxa_aprendizado * gradiente_bias


print("\nPesos finais:")
print(pesos)

print("Bias final:")
print(bias)

print("Loss final:")
print(historico_loss[-1])


# testando o modelo
probabilidades = sigmoid(
    X_teste @ pesos + bias
)

previsoes_teste = (
    probabilidades >= 0.5
).astype(int)

acuracia = np.mean(
    previsoes_teste == y_teste
)

print(
    "\nAcuracia no teste:",
    f"{acuracia * 100:.2f}%"
)


# curva de aprendizado
plt.figure(figsize=(8, 5))

plt.plot(
    range(epocas),
    historico_loss
)

plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.title("Curva de aprendizado")
plt.grid()

plt.show()


# fronteira de decisao
x1_min = X_treino[:, 0].min() - 1
x1_max = X_treino[:, 0].max() + 1

x2_min = X_treino[:, 1].min() - 1
x2_max = X_treino[:, 1].max() + 1

x1, x2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)

pontos = np.c_[
    x1.ravel(),
    x2.ravel()
]

z = sigmoid(
    pontos @ pesos + bias
)

z = z.reshape(x1.shape)

plt.figure(figsize=(8, 6))

plt.contour(
    x1,
    x2,
    z,
    levels=[0.5]
)

plt.scatter(
    X_teste[:, 0],
    X_teste[:, 1],
    c=y_teste
)

plt.xlabel(
    dados.feature_names[indices[0]]
)

plt.ylabel(
    dados.feature_names[indices[1]]
)

plt.title(
    "Fronteira de decisao"
)

plt.show()
