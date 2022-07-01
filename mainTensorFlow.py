import gym
import sys
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
import modelos

def optimize_model():
    funcionPerdida = keras.losses.Huber()
    estados, acciones, recompensas, estados_sig, dones = memoria.sample(BATCH_SIZE)
    dones_tensor = tensorflow.convert_to_tensor(dones)

    y_objetivo = red_objetivo.predict(estados_sig, verbose=0)
    y_objetivo = recompensas + gamma*tensorflow.reduce_max(y_objetivo, axis=1)
    y_objetivo = y_objetivo * (1-dones_tensor) - dones_tensor
    
    mascara = tensorflow.one_hot(acciones, NUM_ACTIONS)
    
    with tensorflow.GradientTape() as cinta:
        Qvalues = red_politica(estados)
        
        y_predicha = tensorflow.reduce_sum(tensorflow.multiply(Qvalues, mascara), axis = 1)
        loss = funcionPerdida(y_predicha, y_objetivo)

    gradientes = cinta.gradient(loss, red_politica.trainable_variables)
    optimizer.apply_gradients(zip(gradientes, red_politica.trainable_variables))
    
def partidasPrueba(numPartidas):
    memoriaEstadosPrueba = []
    while True:
        estadoInicial = np.array(env.reset())
        for i in count():
            accion = env.action_space.sample()
            estado_sig, _, done, _ = env.step(accion)
            estado_sig = np.array(estado_sig)
            
            if np.random.rand(1)[0] > 0.3: memoriaEstadosPrueba.append(estadoInicial)
            estadoInicial = estado_sig
            if done:    break
        if len(memoriaEstadosPrueba) >= numPartidas: break
    return memoriaEstadosPrueba

def updateModel():
    red_objetivo.set_weights(red_politica.get_weights())
    red_objetivo.save("DQN_Atari/Resultados/{}/{}/redObjetivo.hdf5".format(RED, current_time))
    red_politica.save("DQN_Atari/Resultados/{}/{}/redPolitica.hdf5".format(RED, current_time))
    file = open("DQN_Atari/Resultados/{}/{}/epsilon.txt".format(RED, current_time), "a")
    file.write(str(epsilon))
    file.close()

def dibujaGraficaQ(estado):
    print("Dibujo graficas")
    with open('DQN_Atari/EstadosPrueba/{}.memo'.format(estado), 'rb') as f:
            memoriaEstadosEval = pickle.load(f)

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
    for i in range(100):
        recompensaPorPartida = 0
        for vidas in range(5):
            estadoInicial = np.array(env.reset())
            while True:
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

    puntPartidas.append(mediaPuntuaciones)

    plt.plot(puntPartidas)
    plt.savefig("DQN_Atari/Resultados/{}/{}/graficaPuntuaciones.jpg".format(RED, current_time))
    plt.close()

    with open("DQN_Atari/Resultados/{}/{}/valoresPuntuaciones.st".format(RED, current_time), 'wb') as f:
        pickle.dump(mediaPuntuaciones, f)

def entrena():
    global epsilon
    
    for epoca in range(MAX_EPOCH):
        updates = 0
        while True:
            estadoInicial = np.array(env.reset())

            for i in count():
                if np.random.rand(1)[0] > epsilon:
                    accion = env.action_space.sample()
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

                if len(memoria) >= BATCH_SIZE:
                    optimize_model()
                    updates+=1
                if updates % UPDATES_TO_UP_MODEL == 0:
                    updateModel()
                if done: break

            if updates >= UPDATES_TO_EPOCH:
                dibujaGraficaQ("estados1")
                juegaPartidas()
                break
        print("Época {} acabada".format(epoca))

if __name__ == '__main__':
    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
    tensorflow.config.experimental.set_memory_growth(physical_devices[1], True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    medidaTotal = []
    puntPartidas = []
    memoria = Memory(100000)
    seed = 42
    UPDATES_TO_EPOCH = 20000
    UPDATES_TO_UP_MODEL = 10000
    BATCH_SIZE = 32
    EPS_START = 1
    EPS_DECAY = 0.0000009
    EPS_MIN = 0.1
    epsilon = EPS_START
    gamma = 0.99
    K=4
    MAX_EPOCH = 200
    NUM_ACTIONS = 4
    RED = sys.argv[1]

    """
        Inicializacion del entorno:
        Este entorno usa wrappers que nos sirven por ejemplo para que las acciones se
        ejecuten 4 veces sin necesidad de un bucle o por ejemplo nos devuelve directamente las imagenes recortadas.
    """
    env = make_atari("BreakoutNoFrameskip-v4")
        
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)

    with tensorflow.device('/device:GPU:1'):
        if RED == "Base":
            red_politica = modelos.crear_modelo2(NUM_ACTIONS)
            red_objetivo = modelos.crear_modelo2(NUM_ACTIONS)
        elif RED == "Dueling":
            red_politica = modelos.crear_modelo2_Dueling(NUM_ACTIONS)
            red_objetivo = modelos.crear_modelo2_Dueling(NUM_ACTIONS)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        print("Inicializacion acabada")

        


        print("Partidas de prueba acabadas")
        print("Comenzando entrenamiento")
        entrena()
        print("Entrenamiento acabado")

