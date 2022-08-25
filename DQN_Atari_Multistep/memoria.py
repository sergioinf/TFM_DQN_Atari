from collections import deque
import numpy as np


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size, gamma, nSteps = 1):
        indices     = np.random.choice(len(self.memory), batch_size, replace=False)
        states      = []
        actions     = []
        next_states = []
        rewards     = []
        dones       = []
        gammas      = []
        

        for idx in indices:
            actualnSteps = nSteps - 1 #Ponemos el -1 para ajustarnos a los indices de las listas que empiezan en 0
            states.append(self.memory[idx][0])
            actions.append(self.memory[idx][1])

            if idx+actualnSteps >= len(self.memory):
                actualnSteps-=((idx+actualnSteps)-(len(self.memory)-1))

            recompensa = 0
            for elevado in range(actualnSteps + 1):
                recompensa+= (gamma**elevado) * self.memory[idx+elevado][2]
                if self.memory[idx+elevado][4] == True or self.memory[idx+elevado][4] == 3:
                    actualnSteps = elevado
                    break
            rewards.append(recompensa)
            next_states.append(self.memory[idx+actualnSteps][3])
            if self.memory[idx+actualnSteps][4] == 3:
                dones.append(0.0)
            else:
                dones.append(self.memory[idx+actualnSteps][4])
            gammas.append(gamma**(actualnSteps+1))
                

        return np.array(states), actions, rewards, np.array(next_states), dones, gammas

    def __len__(self):
        return len(self.memory)