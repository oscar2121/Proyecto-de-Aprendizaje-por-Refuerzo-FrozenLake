# Proyecto de Aprendizaje por Refuerzo: Q-Learning Tabular en FrozenLake-v1

Repositorio académico y experimental correspondiente a la práctica de **Reinforcement Learning (Aprendizaje por Refuerzo)** para la resolución de entornos estocásticos bajo el marco de los **Procesos de Decisión de Markov (MDP)**.

## 📌 Datos de la Práctica
- **Estudiante:** Oscar
- **Entorno de Simulación:** Gymnasium `FrozenLake-v1` (`is_slippery=True`)
- **Algoritmo:** Q-Learning Tabular (Diferencia Temporal TD(0), Model-Free, Off-Policy)
- **Estrategia de Exploración:** $\varepsilon$-Greedy con decaimiento multiplicativo exponencial
- **Repositorio:** [https://github.com/oscar2121/Proyecto-de-Aprendizaje-por-Refuerzo-FrozenLake](https://github.com/oscar2121/Proyecto-de-Aprendizaje-por-Refuerzo-FrozenLake)

---

## 🎯 Fundamentos Teóricos
El objetivo del agente es hallar una política óptima $\pi^*(s)$ que maximice el retorno esperado acumulado:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### Regla de Actualización de Bellman (Q-Learning):
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

En el modo `is_slippery=True`, la superficie de hielo produce una probabilidad de deslizamiento estocástico:
- **1/3:** Dirección seleccionada por el agente.
- **1/3:** Desvío ortogonal a la izquierda.
- **1/3:** Desvío ortogonal a la derecha.

---

## ⚙️ Hiperparámetros del Experimento
| Hiperparámetro | Valor | Descripción |
| :--- | :---: | :--- |
| **Episodios de Entrenamiento** | 10,000 | Ciclos completos de interacción y aprendizaje |
| **Pasos Máximos por Episodio** | 100 | Límite de corte temporal |
| **Tasa de Aprendizaje ($\alpha$)** | 0.8 | Ponderación del error de diferencia temporal |
| **Factor de Descuento ($\gamma$)** | 0.95 | Ponderación de recompensas a largo plazo |
| **$\varepsilon$ Inicial** | 1.0 | Exploración pura (100% aleatoria) al inicio |
| **$\varepsilon$ Mínimo** | 0.01 | Piso de exploración residual adaptativa (1%) |
| **Factor de Decaimiento ($\delta$)** | 0.9995 | Atenuación multiplicativa por episodio |

---

## 📊 Resultados Obtenidos
- **Tasa de Éxito en Evaluación Determinista ($\varepsilon = 0$ en 1,000 episodios):** **59.40%** (dentro del rango óptimo teórico de FrozenLake resbaladizo).
- **Pasos promedio por episodio:** **39.23 pasos**.
- **Comportamiento aprendido:** El agente desarrolló estrategias defensivas autónomas, navegando pegado a los bordes y utilizando paredes para frenar deslizamientos involuntarios hacia los agujeros letales.

---

## 📁 Estructura del Proyecto
- `Informe_Reinforcement_Learning_FrozenLake.docx`: Documento editable oficial en Word bajo normas APA 7ma edición con títulos centrados en negro puro.
- `Informe_Reinforcement_Learning_FrozenLake.pdf`: Informe compilado oficial en PDF (7 páginas) con figuras de alta resolución incrustadas.
- `Reinforcement_Learning_FrozenLake.ipynb`: Cuaderno Jupyter interactivo para Google Colab.
- `ejecutar_rl.py`: Script modular de entrenamiento, evaluación y generación de figuras.
- `aprendisaje_original.py`: Script base original de la práctica.
- `fig1_entorno_frozenlake.png`: Diagrama del espacio de estados 4x4.
- `fig2_curva_recompensas.png`: Curvas de convergencia y media móvil.
- `fig3_decaimiento_epsilon.png`: Dinámica del parámetro $\varepsilon$.
- `fig4_qtable_politica.png`: Heatmap de la función valor $V(s)$ y política $\pi^*(s)$.
- `metricas_experimento.json`: Registro cuantitativo de la Q-Table y resultados.
