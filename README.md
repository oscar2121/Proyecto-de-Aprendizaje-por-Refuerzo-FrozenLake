import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Definición de la Clase del Agente
class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon_start, epsilon_min, epsilon_decay):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Inicializamos la Q-Table con ceros
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        # Estrategia Epsilon-Greedy encapsulada
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # Exploración
        return np.argmax(self.q_table[state])        # Explotación

    def learn(self, state, action, reward, next_state):
        # Ecuación de Bellman
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Cálculo del nuevo valor Q
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = new_value

    def decay_epsilon(self, episode):
        # Reducción exponencial de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 2. Configuración del Entorno y Entrenamiento
env = gym.make('FrozenLake-v1', render_mode=None, is_slippery=True)

# Hiperparámetros (Ligeramente ajustados para diferenciar)
EPISODES = 10000 # Puedes subir a 15000 si quieres asegurar convergencia
MAX_STEPS = 100
LEARNING_RATE = 0.8
GAMMA = 0.95     # Factor de descuento más alto para pensar más a largo plazo
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995 # Decaimiento multiplicativo (diferente a la fórmula anterior)

# Instanciamos al Agente
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY
)

rewards_history = []

print(" Iniciando entrenamiento del Agente...")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(MAX_STEPS):
        # 1. El agente decide
        action = agent.get_action(state)
        
        # 2. El ambiente reacciona
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # 3. El agente aprende
        agent.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            break
            
    agent.decay_epsilon(episode)
    rewards_history.append(total_reward)
    
    if (episode + 1) % 1000 == 0:
        print(f"Episodio {episode+1}/{EPISODES} - Epsilon: {agent.epsilon:.4f} - Última recompensa: {total_reward}")

print(" Entrenamiento completado.")

# 3. Visualización Avanzada (Media Móvil)
def plot_moving_average(data, window_size=100):
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg, label=f'Media Móvil ({window_size} episodios)', color='blue')
    plt.title('Progreso del Aprendizaje (Suavizado)')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Promedio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

plot_moving_average(rewards_history)

# 4. Demostración Final (Visual)
print("\n🎥 Demostración del Agente Entrenado:")
env_visual = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)
state, _ = env_visual.reset()
done = False

while not done:
    # Le decimos training=False para que no explore, solo explote lo aprendido
    action = agent.get_action(state, training=False)
    state, reward, terminated, truncated, _ = env_visual.step(action)
    env_visual.render()
    
    if terminated or truncated:
        done = True
        if reward == 1:
            print(" ¡El agente llegó a la meta!")
        else:
            print(" El agente cayó en un agujero.")

env_visual.close()# Proyecto-de-Aprendizaje-por-Refuerzo-FrozenLake
