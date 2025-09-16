# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.


## POLICY IMPROVEMENT FUNCTION
#### Name:- Ajay Aswin M
#### Register Number:- 212222240005
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = np.argmax(Q, axis=1)

    return new_pi

def callable_policy(pi_array):
    return lambda s: pi_array[s]

pi_2_array = policy_improvement(V1, P)
pi_2 = callable_policy(pi_2_array)

print("Name:  AJAY ASWIN M   ")
print("Register Number:212222240005         ")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)


```
## POLICY ITERATION FUNCTION
#### Name:- Ajay Aswin M
#### Register Number:- 212222240005
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    # Write your code here for policy iteration
    n_states = len(P)
    n_actions = len(P[0])

    # Initialize a random policy
    pi = np.random.randint(0, n_actions, n_states)

    while True:
        # Policy Evaluation
        V = np.zeros(n_states, dtype=np.float64)
        while True:
            prev_V = V.copy()
            for s in range(n_states):
                v = 0
                action = pi[s]
                for prob, next_state, reward, done in P[s][action]:
                    v += prob * (reward + gamma * prev_V[next_state] * (not done))
                V[s] = v
            if np.max(np.abs(prev_V - V)) < theta:
                break

        # Policy Improvement
        new_pi = np.zeros(n_states, dtype=np.int64)
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            new_pi[s] = np.argmax(Q[s])

        # Check for policy convergence
        if np.array_equal(pi, new_pi):
            break
        pi = new_pi

    return V, callable_policy(pi)
optimal_V, optimal_pi = policy_iteration(P)


print("Name:   AJAY ASWIN M   ")
print("Register Number:  212222240005       ")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)






```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

#### Policy
<img width="558" height="190" alt="image" src="https://github.com/user-attachments/assets/d3765b97-ef33-4029-88b2-e42d019e2015" />


#### Value function
<img width="602" height="183" alt="image" src="https://github.com/user-attachments/assets/2757e56f-eda8-40cf-8485-9a2789ff334e" />


#### success rate
<img width="736" height="76" alt="image" src="https://github.com/user-attachments/assets/4b06590f-6392-474e-9cfc-97451c42907b" />

</br>

### 2. Policy, Value function and success rate for the Improved Policy

#### Policy
<img width="614" height="197" alt="image" src="https://github.com/user-attachments/assets/9aa83da7-b5e5-4b8d-894c-cec9fad74bd1" />

#### Value function
<img width="635" height="214" alt="image" src="https://github.com/user-attachments/assets/8e62bcde-f8c7-4392-a6c4-59c1380841ca" />


#### success rate
<img width="751" height="53" alt="image" src="https://github.com/user-attachments/assets/5a4f3444-4c2b-4155-8966-7856455c1fbd" />

</br>

### 3. Policy, Value function and success rate after policy iteration

</br>

#### Policy
<img width="598" height="209" alt="image" src="https://github.com/user-attachments/assets/827c18b8-6355-47ac-ae69-d0ea66366d93" />

#### success rate
<img width="760" height="61" alt="image" src="https://github.com/user-attachments/assets/52853760-2f3e-40d4-8ede-2e8a4f103d81" />

#### Value function
<img width="1058" height="186" alt="image" src="https://github.com/user-attachments/assets/80ffdf47-359c-42fe-9569-5542b83f6084" />





## RESULT:

Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
