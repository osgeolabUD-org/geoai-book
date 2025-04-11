import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos simulados
area = np.array([100, 150, 200, 250, 300]).reshape(-1, 1)  # Área en m²
avaluo = np.array([80, 100, 120, 150, 170])               # Avalúo en millones

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(area, avluo)

# Predicción
area_pred = np.linspace(90, 310, 100).reshape(-1, 1)
avaluo_pred = modelo.predict(area_pred)

# Gráfica
plt.scatter(area, avluo, color='green', label='Datos reales')
plt.plot(area_pred, avluo_pred, color='black', label='Modelo lineal')
plt.xlabel("Área del terreno (m²)")
plt.ylabel("Avalúo (millones de pesos)")
plt.title("Avalúo catastral vs Área del terreno")
plt.legend()
plt.grid(True)
plt.show()

# Resultados del modelo
print(f"Intercepto (β₀): {modelo.intercept_:.2f} millones")
print(f"Pendiente (β₁): {modelo.coef_[0]:.2f} millones por m²")
