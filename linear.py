import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme
data = pd.read_csv('C:/Users/DELL/Desktop/ödv/house_price.csv')

X1 = data['area'].values / data['area'].max()  # Alanı normalleştir
X2 = data['rooms'].values / data['rooms'].max()  # Oda sayısını normalleştir
Y = data['price'].values  # Hedef değişken (fiyat)


X = np.column_stack((np.ones(X1.shape[0]), X1, X2))

learning_rate = 0.0001
epochs = 10000

# Maliyet fonksiyonu
def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)

# Gradient Descent fonksiyonu
def gradient_descent(X, y, learning_rate, epochs):
    m = len(y)
    theta = np.zeros(X.shape[1])

    cost_history = []
    w0_history = []
    w1_history = []

    for epoch in range(epochs):
        y_pred = X.dot(theta)


        gradient = (1/m) * X.T.dot(y_pred - y)

        theta -= learning_rate * gradient

        cost_loss = compute_cost(X, y, theta)
        cost_history.append(cost_loss)
        w0_history.append(theta[0])
        w1_history.append(theta[1])

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Ağırlıklar: {theta}, Maliyet: {cost_loss:.6f}")

    return theta, cost_history, w0_history, w1_history


def predict_hq(theta, X1_input, X2_input):
    X_input_scaled = np.array([1, X1_input / data['area'].max(), X2_input / data['rooms'].max()])
    predicted_price = X_input_scaled.dot(theta)
    return predicted_price


theta, cost_history, w0_history, w1_history = gradient_descent(X, Y, learning_rate, epochs)

X1_input = float(input("Evin Alanı: "))
X2_input = float(input("Oda Sayısı: "))

predicted_value = predict_hq(theta, X1_input, X2_input)
print(f"Predicted price: ${predicted_value:.2f}")

plt.figure(figsize=(12, 5))

# Hata grafiği
plt.subplot(1, 2, 1)
plt.plot(cost_history, color='blue')
plt.title('Maliyet Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Maliyet')

plt.subplot(1, 2, 2)
plt.plot(w0_history, color='red', label='Q0 ')
plt.plot(w1_history, color='green', label='Q1 (Area Ağırlığı)')
plt.title('Ağırlıkların Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Ağırlık')
plt.legend()

plt.tight_layout()
plt.show()
