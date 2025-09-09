# POLICY ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in policy iteration algorithm
</br>
</br>

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
<img width="597" height="195" alt="image" src="https://github.com/user-attachments/assets/46b96126-f9e1-4d5e-b60a-a691ef7389fe" />


#### Value function
<img width="569" height="180" alt="image" src="https://github.com/user-attachments/assets/d79f3eff-1d6d-4400-9ac6-22b471f009a4" />


#### success rate
<img width="719" height="75" alt="image" src="https://github.com/user-attachments/assets/516bad84-723d-498c-89cd-2c11b9c057bd" />

</br>

### 2. Policy, Value function and success rate for the Improved Policy

#### Policy
<img width="659" height="191" alt="image" src="https://github.com/user-attachments/assets/640b339b-fe79-4c36-ad6c-34f08585b8b1" />

#### Value function
<img width="651" height="180" alt="image" src="https://github.com/user-attachments/assets/51b3aab7-f951-4e4e-93ac-e2d8e9d3b915" />


#### success rate
<img width="746" height="61" alt="image" src="https://github.com/user-attachments/assets/b1c3763f-26a3-43b8-ad11-c62aafecd55d" />

</br>

### 3. Policy, Value function and success rate after policy iteration

</br>


## RESULT:

Write your result here
