import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Configuración de estilo visual para gráficos académicos (estilo sobrio, fondos limpios)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Semilla para reproducibilidad
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.8, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * next_max
        td_error = td_target - old_value
        self.q_table[state, action] = old_value + self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_experiment():
    print("Iniciando entorno FrozenLake-v1 (is_slippery=True)...")
    env = gym.make("FrozenLake-v1", is_slippery=True)
    
    EPISODES = 10000
    MAX_STEPS = 100
    LEARNING_RATE = 0.8
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.9995

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
    epsilon_history = []
    steps_history = []

    print(f"Entrenando agente durante {EPISODES} episodios...")
    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        steps_history.append(steps)

        if (episode + 1) % 2000 == 0:
            recent_win_rate = np.mean(rewards_history[-100:]) * 100
            print(f"Episodio {episode+1}/{EPISODES} | Epsilon: {agent.epsilon:.4f} | WinRate ultimos 100: {recent_win_rate:.1f}%")

    print("\nEvaluando agente entrenado (politica determinista epsilon=0, 1000 episodios)...")
    eval_episodes = 1000
    eval_rewards = []
    eval_steps = []

    for ep in range(eval_episodes):
        state, _ = env.reset(seed=100000 + ep)
        tot_r = 0
        st = 0
        for _ in range(MAX_STEPS):
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            tot_r += reward
            st += 1
            if terminated or truncated:
                break
        eval_rewards.append(tot_r)
        eval_steps.append(st)

    eval_win_rate = float(np.mean(eval_rewards) * 100)
    avg_steps_eval = float(np.mean(eval_steps))
    print(f"Resultado de evaluacion final:")
    print(f" - Tasa de exito (Win Rate): {eval_win_rate:.2f}%")
    print(f" - Pasos promedio por episodio: {avg_steps_eval:.2f}")

    # Guardar métricas en JSON
    metrics = {
        "entorno": "FrozenLake-v1 (is_slippery=True)",
        "episodios_entrenamiento": EPISODES,
        "max_pasos": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "epsilon_inicial": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "eval_episodios": eval_episodes,
        "eval_win_rate_porcentaje": eval_win_rate,
        "eval_pasos_promedio": avg_steps_eval,
        "q_table": agent.q_table.tolist()
    }

    with open("metricas_experimento.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------------------------------------------
    # GENERACIÓN DE FIGURAS ACADÉMICAS
    # -------------------------------------------------------------

    # FIGURA 1: Diagrama del Entorno FrozenLake 4x4 y Dinámica Estocástica
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    grid_desc = [
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G']
    ]
    color_map = {
        'S': '#E2E8F0',  # Inicio (Gris suave)
        'F': '#F8FAFC',  # Superficie helada segura
        'H': '#CBD5E1',  # Agujero (Gris oscuro)
        'G': '#94A3B8'   # Meta (Gris medio)
    }

    for r in range(4):
        for c in range(4):
            val = grid_desc[r][c]
            state_idx = r * 4 + c
            rect = plt.Rectangle((c, 3 - r), 1, 1, facecolor=color_map[val], edgecolor='#000000', linewidth=1.5)
            ax.add_patch(rect)
            
            # Etiquetas
            label = "INICIO (S)" if val == 'S' else ("META (G)" if val == 'G' else ("HOYO (H)" if val == 'H' else "HIELO (F)"))
            ax.text(c + 0.5, 3 - r + 0.65, f"Estado {state_idx}", ha='center', va='center', fontsize=9, fontweight='bold', color='#000000')
            ax.text(c + 0.5, 3 - r + 0.35, label, ha='center', va='center', fontsize=8, color='#333333')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Figura 1. Representación del Espacio de Estados en FrozenLake 4x4\n(Dinámica Estocástica: 1/3 Dirección Deseada, 2/3 Deslizamiento Ortogonal)",
                 fontsize=11, fontweight='bold', pad=15, color='#000000')
    plt.tight_layout()
    plt.savefig("fig1_entorno_frozenlake.png")
    plt.close()

    # FIGURA 2: Curva de Convergencia de Recompensas (Media Móvil)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    w_size = 100
    moving_avg_100 = np.convolve(rewards_history, np.ones(w_size)/w_size, mode="valid")
    w_size_500 = 500
    moving_avg_500 = np.convolve(rewards_history, np.ones(w_size_500)/w_size_500, mode="valid")

    ax.plot(moving_avg_100, label="Media móvil (Ventana = 100 episodios)", color="#666666", linewidth=1.0, alpha=0.7)
    ax.plot(moving_avg_500, label="Tendencia global (Ventana = 500 episodios)", color="#000000", linewidth=2.0)
    ax.axhline(y=eval_win_rate/100.0, color="#000000", linestyle="--", linewidth=1.2, label=f"Tasa de éxito final eval ({eval_win_rate:.1f}%)")

    ax.set_title("Figura 2. Curva de Convergencia del Rendimiento (Recompensa Acumulada)", fontsize=11, fontweight='bold', color='#000000', pad=12)
    ax.set_xlabel("Episodio de Entrenamiento", fontsize=10, fontweight='bold', color='#000000')
    ax.set_ylabel("Tasa de Éxito / Recompensa Promedio", fontsize=10, fontweight='bold', color='#000000')
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6, color="#888888")
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#333333', fontsize=9)
    plt.tight_layout()
    plt.savefig("fig2_curva_recompensas.png")
    plt.close()

    # FIGURA 3: Decaimiento de Epsilon (Dilema Exploración vs Explotación)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    ax.plot(epsilon_history, color="#000000", linewidth=1.8, label=r"Tasa de Exploración $\epsilon(t) = \max(\epsilon_{min}, \epsilon_0 \cdot \delta^t)$")
    ax.axvline(x=np.where(np.array(epsilon_history) <= 0.05)[0][0], color="#555555", linestyle="--", label="Umbral $\epsilon \leq 0.05$ (~6000 episodios)")
    ax.set_title(r"Figura 3. Decaimiento Exponencial del Parámetro Epsilon ($\epsilon$)", fontsize=11, fontweight='bold', color='#000000', pad=12)
    ax.set_xlabel("Episodio", fontsize=10, fontweight='bold', color='#000000')
    ax.set_ylabel(r"Probabilidad de Exploración ($\epsilon$)", fontsize=10, fontweight='bold', color='#000000')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6, color="#888888")
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#333333', fontsize=9)
    plt.tight_layout()
    plt.savefig("fig3_decaimiento_epsilon.png")
    plt.close()

    # FIGURA 4: Heatmap de Q-Table y Política Óptima Resultante
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Subgráfico 1: Heatmap de Valores Máximos V(s) = max_a Q(s,a)
    v_values = np.max(agent.q_table, axis=1).reshape((4, 4))
    im = ax1.imshow(v_values, cmap='Greys', interpolation='nearest', vmin=0, vmax=np.max(v_values))
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r'Valor Máximo Esperado $V(s) = \max_a Q(s, a)$', fontsize=9, color='#000000')

    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            val = v_values[r, c]
            text_color = '#ffffff' if val > (np.max(v_values) * 0.5) else '#000000'
            cell_type = grid_desc[r][c]
            ax1.text(c, r, f"S{idx}\n({cell_type})\n{val:.3f}", ha='center', va='center',
                     fontsize=8, fontweight='bold', color=text_color)

    ax1.set_title(r"(A) Función Valor de Estado $V(s)$", fontsize=10, fontweight='bold', color='#000000')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))

    # Subgráfico 2: Política Óptima pi*(s) con Flechas de Dirección
    # Acciones Gym: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    action_names = {0: 'Izq', 1: 'Abj', 2: 'Der', 3: 'Arr'}

    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            cell_type = grid_desc[r][c]
            bg_color = '#F1F5F9' if cell_type in ['S', 'F'] else ('#E2E8F0' if cell_type == 'G' else '#94A3B8')
            rect = plt.Rectangle((c, 3 - r), 1, 1, facecolor=bg_color, edgecolor='#000000', linewidth=1.2)
            ax2.add_patch(rect)

            if cell_type in ['H']:
                ax2.text(c + 0.5, 3 - r + 0.5, f"HOYO\n(S{idx})", ha='center', va='center', fontsize=9, fontweight='bold', color='#000000')
            elif cell_type == 'G':
                ax2.text(c + 0.5, 3 - r + 0.5, f"META\n(S{idx})", ha='center', va='center', fontsize=9, fontweight='bold', color='#000000')
            else:
                best_action = int(np.argmax(agent.q_table[idx]))
                arrow = action_arrows[best_action]
                act_str = action_names[best_action]
                ax2.text(c + 0.5, 3 - r + 0.65, arrow, ha='center', va='center', fontsize=18, fontweight='bold', color='#000000')
                ax2.text(c + 0.5, 3 - r + 0.25, f"S{idx}: {act_str}", ha='center', va='center', fontsize=8, color='#222222')

    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(r"(B) Política Óptima Aprendida $\pi^*(s)$", fontsize=10, fontweight='bold', color='#000000')

    plt.suptitle("Figura 4. Diagnóstico de la Tabla Q: Valores de Estado y Política Determinista", fontsize=11, fontweight='bold', color='#000000', y=0.98)
    plt.tight_layout()
    plt.savefig("fig4_qtable_politica.png")
    plt.close()

    print("Experimento finalizado exitosamente y graficos generados.")

if __name__ == "__main__":
    run_experiment()
