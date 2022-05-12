import gym
import random
import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import logging
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle as p

env = gym.make('ALE/Breakout-v5')

numAcciones = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training with:", device)


#############################################################
#--------------------------- Logger -------------------------
#############################################################

logging.basicConfig(filename="logFile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)
#############################################################



######################################################################
#------------------- Process the next frame   ------------------------
######################################################################

def process_image(screen = None):
    if screen is None:
        screen = env.render(mode='rgb_array')
    grayimg = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    scaled = cv2.resize(grayimg, (84, 110))
    cropped_image = scaled[26:110, 0:84]

    return cropped_image
######################################################################


#######################################################################################
#--------------------------------------- Memory ---------------------------------------
#######################################################################################


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def save(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        indices     = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states      = []
        actions     = []
        next_states = []
        rewards     = []


        for idx in indices: 
            states.append(self.memory[idx][0])
            actions.append(self.memory[idx][1])
            next_states.append(self.memory[idx][2])
            rewards.append(self.memory[idx][3])
        
        return states, actions, rewards, next_states
        #return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#######################################################################################


memory = Memory(1000000)
BUFFER_CAP = 4
BATCH_SIZE = 32
EPS_START = 1
EPS_DECAY = 0.0000001
EPS_MIN = 0.1
epsilon = EPS_START
gamma = 0.99
K = 4



#######################################################################################
#------------------------------- Memory Optimizer -------------------------------------
#######################################################################################

def optimize_model():

    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample(BATCH_SIZE)
    estado, accion, recompensa, estado_sig = batch

    estado = np.array(estado)
    estado_sig = np.array(estado_sig)

    try:
        listaNones = np.where(estado_sig == None)
    except:
        listaNones = []

    tensor_accion       = torch.Tensor(accion).to(device)
    tensor_recompensa   = torch.Tensor(recompensa).to(device)

    estado_sig[listaNones] = estado[listaNones]

    Qvalues = [red_politica(e).max(1)[0].item() for e in estado]
    
    QpValues = [red_objetivo(e).max(1)[0].item() for e in estado_sig]
    QpValues = np.array(QpValues)
    QpValues[listaNones] = 0.0


    Qvalues = torch.Tensor(Qvalues).to(device)
    Qvalues.requires_grad_()
    QpValues = torch.Tensor(QpValues).to(device)
    QpValues.requires_grad_()

    valorEsperado = QpValues * gamma + tensor_recompensa
    

    Qvalues.retain_grad()
    valorEsperado.retain_grad()
    
    loss = nn.MSELoss() 
    output = loss(Qvalues, valorEsperado)

    medida = np.mean([i.item() for i in valorEsperado])

    optimizer.zero_grad()
    output.backward()
    optimizer.step()

    return medida

########################################################################################



################################################################################################################
#----------------------------------------- Estructura de la red ------------------------------------------------
################################################################################################################

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        """Para calcular correctamente la salida, tenemos que linealizarla, esto depende de las dimensiones
        de las imagenes de entrada y de los parámetros introducidos"""
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
        self.linear_input_size = convw * convh * 32

        self.hl = nn.Linear(self.linear_input_size, 256)
        self.ol = nn.Linear(256, outputs)

    """Devuelve un vector con el valor de las acciones posibles"""
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, self.linear_input_size)

        return self.ol(self.hl(x))

#####################################################################################################################



##########################################################################
#-------------------- Inicialización de redes ----------------------------
##########################################################################
screen_height, screen_width = 84, 84

red_politica = DQN(screen_height, screen_width, numAcciones).to(device)
red_objetivo = DQN(screen_height, screen_width, numAcciones).to(device)
red_objetivo.load_state_dict(red_politica.state_dict())
red_objetivo.eval()
scoreList = []
optimizer = optim.RMSprop(red_politica.parameters())
print("Redes inicializadas")

# #----------------------- Carga red de fichero ------------------#
# pickle_in = open('listaScore','rb')                       #
# scoreList = p.load(pickle_in)                                   #
# pickle_in.close()                                               #
#                                                                 #

# # pickle_in = open('Memory','rb')                           #
# # memory.memory = p.load(pickle_in)                               #
# # pickle_in.close()                                               #
#                                                                 #
#                                                                 #
# red_politica.load_state_dict(torch.load('RedPolitica.pt'))#
# red_objetivo.load_state_dict(torch.load('RedObjetivo.pt'))#
# #---------------------------------------------------------------#



############################################################################################
#-------------------------------------- Selector de acciones -------------------------------
############################################################################################
def action_selection(state):
    global epsilon
    
    epsilon = epsilon-EPS_DECAY
    if epsilon < EPS_MIN:
        epsilon = EPS_MIN
    

    if random.randint(0, 100)/100 < epsilon:
        return random.randrange(numAcciones)
    else:
        with torch.no_grad():
            return  red_politica(state).max(1)[1]

##############################################################################################




############################################################################################
#-------------------------------------- Bucle de entrenamiento -------------------------------
############################################################################################
print("Comienzo del entrenamiento:")

episodios = 1000000
medidaTotal = []

nombrePlot = time.time()
for partida in range(episodios):
    env.reset()
    ImageBuffer = []
    for frame in range(K):
        ImageBuffer.append(process_image())
        
    estado = torch.Tensor(ImageBuffer)

    score = 0
    medidaPartida = []
    for j in count():
        accion = action_selection(estado) 

        recompensa=0
        sigEstado = []
        for frame in range(K):
            sigImg, r, done, _ = env.step(accion)
            sigEstado.append(process_image(sigImg))
            recompensa+=r
        
        if not done:
            sigEstado = torch.Tensor(sigEstado)
        else:
            sigEstado = None
            
        memory.save(estado, accion, sigEstado, recompensa)

        estado = sigEstado
        score+=recompensa
        if j%100 == 0 and j != 0 :
            medidaPartida.append(optimize_model())
            red_objetivo.load_state_dict(red_politica.state_dict())

        if done:
            medidaPartida.append(optimize_model())
            scoreList.append(score)
            break
    
    medidaTotal.append(np.mean(medidaPartida))
    grafico = plt.plot(medidaTotal)
    plt.savefig(str(nombrePlot)+".jpg")
    plt.close()


    if partida % 100 == 0:
        logger.debug("Partida {} acabada.".format(partida))
        torch.save(red_objetivo.state_dict(), "RedObjetivo.pt")
        torch.save(red_politica.state_dict(), "RedPolitica.pt")

        outputFile = open('listaScore', 'wb')
        p.dump(scoreList, outputFile)
        outputFile.close()

logger.debug("Entranamiento de 1M de partidas acabado")