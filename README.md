# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the Monte carlo control algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

### States
- **5 Terminal States**:
  - `G` (Goal): The state the agent aims to reach.
  - `H` (Hole): A hazardous state that the agent must avoid at all costs.
- **11 Non-terminal States**:
  - `S` (Starting state): The initial position of the agent.
  - Intermediate states: Grid cells forming a layout that the agent must traverse.

### Actions
The agent can take 4 actions in each state:
- `LEFT`
- `RIGHT`
- `UP` 
- `DOWN`

### Transition Probabilities
The environment is stochastic, meaning that the outcome of an action is not always certain.
- **33.33%** chance of moving in the intended direction.
- **66.66%** chance of moving in a orthogonal directions.

This uncertainty adds complexity to the agent's navigation.

### Rewards
- `+1` for reaching the goal state(G).
- 0 reward for all other states, including the starting state (S) and intermediate states.

### Episode Termination
The episode terminates when the agent reaches the goal state (G) or falls into a hole (H).

### GRAPHICAL REPRESENTATION
![275603322-b0556bd7-2dc8-4066-8c34-6e4141cc25b1](https://github.com/Pravinrajj/monte-carlo-control/assets/117917674/7a54df6e-1d5b-4171-b281-5b4a4c8af7c8)

## MONTE CARLO CONTROL ALGORITHM
1. Initialize the state value function V(s) and the policy π(s) arbitrarily.
2. Generate an episode using π(s) and store the state, action, and reward sequence.
3. For each state s appearing in the episode:
    * G ← return following the first occurrence of s
    * Append G to Returns(s)
    * V(s) ← average(Returns(s))
4. For each state s in the episode:
    * π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')
5. Repeat steps 2-4 until the policy converges.
6. Use the function `decay_schedule` to decay the value of epsilon and alpha.
7. Use the function `gen_traj` to generate a trajectory.
8. Use the function `tqdm` to display the progress bar.
9. After the policy converges, use the function `np.argmax` to find the optimal policy. The function takes the following arguments:
    * `Q`: The Q-table.
    * `axis`: The axis along which to find the maximum value.

## ALGORITHM
### STEP 1
Initialize the agent's knowledge of the environment, which includes the Q-values, state-value function, and policy.

### STEP 2
The agent explores the environment, taking actions and observing the rewards and next states. This process is repeated until an episode ends.

### STEP 3
For each step in the episode, the agent updates its Q-value estimate for the state-action pair it took. The update is based on the reward received and the value of the next state.

### STEP 4
The agent updates its policy to select the action with the highest Q-value for each state.

### STEP 5
The agent repeats steps 2-4 for a specified number of episodes or until it converges to a good policy.

### STEP 6
The agent returns the optimal Q-values, state-value function, and policy.

## MONTE CARLO CONTROL ALGORITHM
```
Developed By: PRAVINRAJJ G.K
Register no: 212222240080
```
```py
import numpy as np
from tqdm import tqdm

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    # Get the number of states and actions
    nS, nA = env.observation_space.n, env.action_space.n

    # Create an array for discounting
    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n))

    # Create schedules for alpha and epsilon decay
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Initialize Q-table and tracking array
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        # Generate a trajectory
        traj = gen_traj(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    # Calculate the value function and policy
    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```
## PROGRAM TO EVALUATE THE POLICY
```py
# number of episodes = 450000
import random
import numpy as np

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    results = []

    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            action = pi[state]
            state, _, done, _ = env.step(action)
            steps += 1
        results.append(state == goal_state)

    success_rate = np.sum(results) / len(results)
    return success_rate

def mean_return(env, pi, n_episodes=100, max_steps=200, seed=123):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    results = []

    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        returns = 0.0
        while not done and steps < max_steps:
            action = pi[state]
            state, reward, done, _ = env.step(action)
            returns += reward
            steps += 1
        results.append(returns)

    average_return = np.mean(results)
    return average_return

def results(env, optimal_pi, goal_state, seed=123):
    success_rate = probability_success(env, optimal_pi, goal_state=goal_state, seed=seed)
    avg_return = mean_return(env, optimal_pi, seed=seed)
    
    print(f'Reaches goal {success_rate:.2%}. 
  			Obtains an average undiscounted return of: {avg_return:.4f}.')

goal_state = 15
results(env, optimal_pi, goal_state=goal_state) 

```
## OUTPUT:
![280755067-de865f54-3ef4-414b-91fa-b5872202152d](https://github.com/Pravinrajj/monte-carlo-control/assets/117917674/d24dcecd-72d6-49c9-afa2-f223c0ccfb48)
![280755093-383f5977-817c-446f-ae43-cdabc5a8b8ec](https://github.com/Pravinrajj/monte-carlo-control/assets/117917674/339e770d-eaf3-4585-8f5a-1ae1ffc18156)

## RESULT:
Thus, Python program is developed to find the optimal policy for the given RL environment using the Monte Carlo algorithm.
