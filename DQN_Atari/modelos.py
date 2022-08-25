import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def crear_modelo(NUM_ACTIONS):
    entrada = layers.Input(shape = (84,84,4))

    conv1 = layers.Conv2D(16, 8, strides=4, activation = 'relu')(entrada)
    conv2 = layers.Conv2D(32, 4, strides=2, activation = 'relu')(conv1)
    layer3 = layers.Flatten()(conv2)
        
    fc1 = layers.Dense(256, activation='relu')(layer3)
    #fc1 = layers.Dense(256)(layer3)
    ou = layers.Dense(NUM_ACTIONS, activation = 'linear')(fc1)

    return keras.Model(inputs=entrada, outputs = ou)

def crear_modelo2(NUM_ACTIONS):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(NUM_ACTIONS, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def crear_modelo_Dueling(NUM_ACTIONS):
    entrada = layers.Input(shape = (84,84,4))

    conv1 = layers.Conv2D(16, 8, strides=4, activation = 'relu')(entrada)
    conv2 = layers.Conv2D(32, 4, strides=2, activation = 'relu')(conv1)
    layer3 = layers.Flatten()(conv2)
        
    fc1 = layers.Dense(256, activation='relu')(layer3)
    advantage = layers.Dense(NUM_ACTIONS, activation = 'linear')(fc1)

    fc2 = layers.Dense(256, activation='relu')(layer3)
    value = layers.Dense(1, activation = 'linear')(fc2)

    output = value + advantage - tf.reduce_mean(advantage)
    
    return keras.Model(inputs=entrada, outputs = output)

def crear_modelo2_Dueling(NUM_ACTIONS):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    advantage = layers.Dense(NUM_ACTIONS, activation="linear")(layer5)

    layer6 = layers.Dense(512, activation="relu")(layer4)
    value = layers.Dense(1, activation="linear")(layer6)

    output = value + advantage - tf.reduce_mean(advantage)

    return keras.Model(inputs=inputs, outputs=output)