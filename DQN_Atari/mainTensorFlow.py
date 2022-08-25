import gym
import sys
import os
from collections import deque
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from memoria import Memory
from itertools import count
import datetime
import pickle
import random
import modelos

def optimize_model():
    estados, acciones, recompensas, estados_sig, dones = memoria.sample(BATCH_SIZE)
    dones_tensor = tensorflow.convert_to_tensor(dones)

    qpvalues = red_objetivo.predict(estados_sig, verbose=0)
    y_objetivo = recompensas + gamma*tensorflow.reduce_max(qpvalues, axis=1)
    y_objetivo = y_objetivo * (1-dones_tensor) - dones_tensor
    
    mascara = tensorflow.one_hot(acciones, NUM_ACTIONS)
    
    with tensorflow.GradientTape() as cinta:
        Qvalues = red_politica(estados)
        
        y_predicha = tensorflow.reduce_sum(tensorflow.multiply(Qvalues, mascara), axis = 1)
        loss = funcionPerdida(y_objetivo, y_predicha)

    gradientes = cinta.gradient(loss, red_politica.trainable_variables)
    optimizer.apply_gradients(zip(gradientes, red_politica.trainable_variables))
    

def updateModel():
    red_objetivo.set_weights(red_politica.get_weights())
    red_objetivo.save("DQN_Atari/Resultados/{}/{}/redObjetivo.hdf5".format(RED, current_time))
    red_politica.save("DQN_Atari/Resultados/{}/{}/redPolitica.hdf5".format(RED, current_time))

    file = open("DQN_Atari/Resultados/{}/{}/epsilon.txt".format(RED, current_time), "a")
    file.write(str(epsilon))
    file.write('\n')
    file.close()

    file = open("DQN_Atari/Resultados/{}/{}/memoria.txt".format(RED, current_time), "a")
    file.write(str(len(memoria)))
    file.write('\n')
    file.close()

def dibujaGraficaQ(estado):
    print("Dibujo graficas")
    with open('DQN_Atari/EstadosPrueba/{}.memo'.format(estado), 'rb') as f:
            memoriaEstadosEval = pickle.load(f)
        
    if not os.path.exists("DQN_Atari/Resultados/{}/{}".format(RED, current_time)):
        os.mkdir("DQN_Atari/Resultados/{}/{}".format(RED, current_time))

    estados = np.array(memoriaEstadosEval)
    Qvalues = red_politica(estados, training = False)
    Qvalues = tensorflow.reduce_max(Qvalues, axis=1)

    media = tensorflow.reduce_mean(Qvalues).numpy()

    medidaTotal.append(media)
    plt.plot(medidaTotal)
    plt.savefig("DQN_Atari/Resultados/{}/{}/graficaConvergencia_{}.jpg".format(RED, current_time, estado))
    plt.close()

    with open("DQN_Atari/Resultados/{}/{}/valoresQ_{}.st".format(RED, current_time, estado), 'wb') as f:
        pickle.dump(medidaTotal, f)

def juegaPartidas():
    print("Juego las partidas")
    puntuacionesTotales = []
    for i in range(20):
        recompensaPorPartida = 0
        for vidas in range(5):
            estadoInicial = np.array(env.reset())
            for i in count():
                estado_tensor = tensorflow.convert_to_tensor(estadoInicial)
                estado_tensor = tensorflow.expand_dims(estadoInicial, 0)

                salida = red_objetivo(estado_tensor, training=False)
                accion = tensorflow.argmax(salida[0]).numpy()

                estado_sig, recompensa, done, _ = env.step(accion)
                recompensaPorPartida+=recompensa
                estado_sig = np.array(estado_sig)

                estadoInicial = estado_sig

                if done:  
                    break
        puntuacionesTotales.append(recompensaPorPartida)
    mediaPuntuaciones = np.array(puntuacionesTotales).mean()
    desviacion = np.std(np.array(puntuacionesTotales))
    desviacionesTipicas.append(desviacion)
    puntPartidas.append(mediaPuntuaciones)

    plt.plot(puntPartidas)
    plt.savefig("DQN_Atari/Resultados/{}/{}/graficaPuntuaciones.jpg".format(RED, current_time))
    plt.close()

    with open("DQN_Atari/Resultados/{}/{}/valoresPuntuaciones.st".format(RED, current_time), 'wb') as f:
        pickle.dump(puntPartidas, f)
    with open("DQN_Atari/Resultados/{}/{}/desviacionesTipicas.st".format(RED, current_time), 'wb') as f:
        pickle.dump(desviacionesTipicas, f)

def entrena():
    global epsilon
    pasos = 0
    juegaPMedia = False

    while True:
        estadoInicial = np.array(env.reset())
        for i in range(1, PASOS_MAX_EPISODIO):
            pasos+=1
            if pasos < 50000 or np.random.rand(1)[0] < epsilon:
                accion = random.randint(0, NUM_ACTIONS-1)
            else:
                estado_tensor = tensorflow.convert_to_tensor(estadoInicial)
                estado_tensor = tensorflow.expand_dims(estadoInicial, 0)
                
                salida = red_politica(estado_tensor, training=False)
                
                accion = tensorflow.argmax(salida[0]).numpy()
            epsilon-=EPS_DECAY

            epsilon = max(epsilon, EPS_MIN)
            estado_sig, recompensa, done, _ = env.step(accion)
            estado_sig = np.array(estado_sig)

            memoria.save(estadoInicial, accion, recompensa, estado_sig, float(done))
            estadoInicial = estado_sig

            if pasos % K == 0 and len(memoria) >= BATCH_SIZE:
                optimize_model()
            if pasos % PASOS_UP_MODEL == 0:
                updateModel()
                dibujaGraficaQ("estados1")
                if juegaPMedia:
                    juegaPartidas()
                    juegaPMedia = False
                else: juegaPMedia = True
            
            if done: break


if __name__ == '__main__':
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
    tensorflow.config.experimental.set_memory_growth(physical_devices[1], True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with tensorflow.device('/device:GPU:1'):
        NUM_ACTIONS = 4
        K=4
        seed = 42
        gamma = 0.99
        EPS_START = 1
        EPS_DECAY = 0.0000009
        EPS_MIN = 0.1
        epsilon = EPS_START
        BATCH_SIZE = 32
        PASOS_MAX_EPISODIO = 10000
        PASOS_UP_MODEL = 5000
        RED = sys.argv[1]


        env = make_atari("BreakoutNoFrameskip-v4")
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        env.seed(seed)

        #with tensorflow.device('/device:GPU:1'):
        if RED == "Base":
            red_politica = modelos.crear_modelo2(NUM_ACTIONS)
            red_objetivo = modelos.crear_modelo2(NUM_ACTIONS)
        elif RED == "Dueling":
            red_politica = modelos.crear_modelo2_Dueling(NUM_ACTIONS)
            red_objetivo = modelos.crear_modelo2_Dueling(NUM_ACTIONS)
            
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        funcionPerdida = keras.losses.Huber()

        medidaTotal = []
        puntPartidas = []
        desviacionesTipicas = []
        memoria = Memory(10000)

        print("Inicializacion acabada")
        print("Comenzando entrenamiento")
        entrena()
        print("Entrenamiento acabado")

