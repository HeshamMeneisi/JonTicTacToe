from __future__ import print_function
import game.game_container as ttt
import argparse, random, os.path, json, sys
import skimage
from skimage import transform, color, exposure
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
import pylab as plt
# import theano

MODEL_NAME = "Jon"          # An AI has to have a name! Also the subdirectory name
MODEL_VERSION = 200         # The version is used for the file name
VERSION_UPDATE = 1200       # New version interval, in seconds
SAVE_INTERVAL = 500         # Save interval in iterations
DUMPING_INTERVAL = 5000     # Experience memory dumping interval in iterations
REPORT_INTERVAL = 100       # Frames before reporting, if not verbose
THROTTLING_PERIOD = 1       # Frames to skip before training again
ACTIONS = 9                 # Number of valid actions
INITIAL_GAMMA = 0.4         # Low confidence in predictions while exploring
# ** A large gamma for a game where positive rewards are usually subsequent can eventually cause an overflow
FINAL_GAMMA = 0.6           # High confidence when perfecting the technique
OBSERVE_PERIOD = 5000       # Frames to observe before training
EXPLORE_PERIOD = 200000     # Iterations over which to anneal EPSILON and grow GAMMA from initial to final
FINAL_EPSILON = 0.01        # Final value of EPSILON
INITIAL_EPSILON = 0.3       # Starting value of EPSILON
MEMORY_SIZE = 50000         # Number of previous transitions to remember
BATCH = 64                  # Size of experiences to train on
DISC_REWARD_DEPTH = 2       # Subsequent states to use in discounted reward, 0 means immediate reward only
GAMMA_DECAY_MUL = 2         # The gamma decay multiplier, a higher value means a faster fall of confidence
FRAMES_PER_ACTION = 1       # The delay before taking another action
LEARNING_RATE = 1e-4        # Our network's learning rate
FRAMES_PER_SAMPLE = 1       # How many frames to stack per sample, good for detecting time-based amounts like velocity
RESTORE_STATE = True        # Whether or not to restore a state if found
RESTORE_MEMORY = True       # Whether or not to restore the experience memory if found
# ** The image size, images are rotated 90 degrees in a matrix so the height is rows and the width is columns
# ** It's better to use a square image because otherwise lines might get jagged or completely disappear while resizing
IMG_ROWS, IMG_COLS = 60, 60

# A map of all action names for logging
AMAP = ['0,0', '0,1', '0,2', '1,0', '1,1', '1,2', '2,0', '2,1', '2,2']

# Paths
MODEL_DIR = ".\\"+MODEL_NAME
SAVE_PATH = lambda: MODEL_DIR+"\\"+str(MODEL_VERSION)
STATE_PATH = MODEL_DIR+"\\state.npy"
DUMPING_PATH = MODEL_DIR+"\\memory"
MEM_STATE_EXT = ".state.npy"
MEM_DATA_EXT = ".dump.npy"

# When training, the state is reset to observing on restart. No memory dumping yet.
STATE_LABELS = ['Testing', 'Observing', 'Exploring', 'Training']

# A state is represented by the FRAMES_PER_SAMPLE starting at STATE_POINTER
STATE_POINTER = 0
STORAGE_SIZE = 0


def display_image(img):
    # This code would display the image identical to the game itself
    # However, orientation and/or mirroring will not affect the training
    # This is mainly because there's no sense of directions in the network when initialized
    # For demonstration, this is similar to how babies see the world upside down for the first few days
    plt.imshow(np.fliplr(skimage.transform.rotate(skimage.exposure.rescale_intensity(
        skimage.transform.resize(img, (IMG_ROWS, IMG_COLS)), out_range=(0, 255)), -90)), cmap='gray')


# Build the model
def build_model():
    print("Building the model...")
    # We can try predicting exactly how the model would understand the game and tune it for the job
    model = Sequential()
    # Detect X/O/E
    cell_sz = (int(IMG_ROWS/3), int(IMG_COLS/3))
    model.add(Convolution2D(3, kernel_size=cell_sz, strides=cell_sz, padding='same',
              input_shape=(IMG_ROWS, IMG_COLS, FRAMES_PER_SAMPLE)))
    model.add(Activation('relu'))
    # Detect patterns of Xs and Os in a 2x2 area, there are a total of 81 possible patterns (3^4)
    model.add(Convolution2D(81, kernel_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    # Deduce action
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("Done building.")
    return model


def save_model(model, save_state=True):
    print("Saving model...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save the weights
    model.save_weights(SAVE_PATH(), overwrite=True)

    # Dump the model structure info for future reference
    with open(MODEL_DIR + "\\model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    print("Saved to: ", SAVE_PATH())

    if save_state:
        # Save the state variables
        np.save(STATE_PATH, (T, MODEL_VERSION, EPSILON, GAMMA))
        print("State saved to: ", STATE_PATH)


def dump_memory():
    print("Dumping the experience memory...")
    np.save(DUMPING_PATH + MEM_STATE_EXT, [STORAGE_SIZE, STATE_POINTER])
    np.save(DUMPING_PATH + MEM_DATA_EXT,
            (FRAME_STORAGE[:, :, :, :STORAGE_SIZE + FRAMES_PER_SAMPLE - 1], ACTIONS_LOG[:STORAGE_SIZE],
             REWARD_LOG[:STORAGE_SIZE], TFLAG_LOG[:STORAGE_SIZE]))
    print("Memory dumped")


def load_model(model, path):
    print("Loading weights for model ", MODEL_NAME, " version ", MODEL_VERSION)
    model.load_weights(path)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("Weights loaded from", path)


def lookup_state():
    global T, MODEL_VERSION, EPSILON, GAMMA
    if os.path.exists(STATE_PATH):
        print("Found a state file, restoring...")
        (T, MODEL_VERSION, EPSILON, GAMMA) = np.load(STATE_PATH)
        T = int(T)+1  # To avoid unnecessary saving when loading a periodic save
        MODEL_VERSION = int(MODEL_VERSION)
        print("State restored", "T =", T, "Version =", MODEL_VERSION, "EPSILON =", EPSILON, "GAMMA =", GAMMA)


def lookup_experience():
    global STORAGE_SIZE, STATE_POINTER, FRAME_STORAGE, ACTIONS_LOG, REWARD_LOG, TFLAG_LOG
    if os.path.exists(DUMPING_PATH + MEM_STATE_EXT) and os.path.exists(DUMPING_PATH + MEM_DATA_EXT):
        print("Found a memory dump, restoring...")
        [STORAGE_SIZE, STATE_POINTER] = np.load(DUMPING_PATH + MEM_STATE_EXT)

        sz = MEMORY_SIZE
        if STORAGE_SIZE > MEMORY_SIZE:
            print("Can only accommodate (", sz / STORAGE_SIZE * 100, "% ) of the stored memory.")
            data = np.load(DUMPING_PATH + MEM_DATA_EXT)
            FRAME_STORAGE = data[0][:, :, :, :sz + FRAMES_PER_SAMPLE - 1]
            ACTIONS_LOG = data[1][:sz]
            REWARD_LOG = data[2][:sz]
            TFLAG_LOG = data[3][:sz]
        else:
            sz = STORAGE_SIZE
            print("The dumped memory will pre-fill (", sz / MEMORY_SIZE * 100, "% ) of the current memory.")
            (FRAME_STORAGE[:, :, :, :sz + FRAMES_PER_SAMPLE - 1], ACTIONS_LOG[:sz], REWARD_LOG[:sz], TFLAG_LOG[:sz]) \
                = np.load(DUMPING_PATH + MEM_DATA_EXT)
        print("Experience restored.")


# Run the selected action and observe the next state and reward from game
def do_action(action):
    global STORAGE_SIZE, STATE_POINTER, FRAME_STORAGE, ACTIONS_LOG, REWARD_LOG, TFLAG_LOG

    # Next state, reward, gameover?, actual action
    x_next_colored, reward, terminal, tact = ttt.step(action)

    if INTERVENTION_WATCH & (tact != action):
        action = tact
        print("User action: ", action)

    if PROG_MODE == 0:
        ACTIONS_LOG[STATE_POINTER] = action
        REWARD_LOG[STATE_POINTER] = reward

    # Preprocess first to remove extra data and highlight contours
    x_next = skimage.color.rgb2gray(x_next_colored)
    x_next = skimage.transform.resize(x_next, (IMG_ROWS, IMG_COLS), mode='constant')
    x_next = skimage.exposure.rescale_intensity(x_next, out_range=(0, 255)).astype(np.ubyte)  # For memory efficiency
    x_next = x_next.reshape(1, IMG_ROWS, IMG_COLS, 1)  # First dimension depth is one for keras
    # Update the pointer to the next state
    STATE_POINTER += 1
    if STATE_POINTER >= MEMORY_SIZE:
        STATE_POINTER = 0
        FRAME_STORAGE[:, :, :, 0:FRAMES_PER_SAMPLE-1] = FRAME_STORAGE[:, :, :, MEMORY_SIZE:]  # Initial frames no longer the previous
    idx = STATE_POINTER + FRAMES_PER_SAMPLE - 1
    FRAME_STORAGE[:, :, :, idx:idx+1] = x_next

    if PROG_MODE == 0:
        TFLAG_LOG[STATE_POINTER] = terminal

    if STORAGE_SIZE < MEMORY_SIZE:
        STORAGE_SIZE += 1

    return reward, terminal, action


def get_state(idx):
    return FRAME_STORAGE[:, :, :, idx:idx + FRAMES_PER_SAMPLE]


def get_state_data(idx):
    global FRAME_STORAGE, ACTIONS_LOG, REWARD_LOG, TFLAG_LOG
    s = get_state(idx)
    t = TFLAG_LOG[idx]
    r = REWARD_LOG[idx]
    a = ACTIONS_LOG[idx]
    return s, t, a, r


def one_iteration(model):
    # Pick a random sample from the replay memory to train on
    experiences = np.random.randint(0, STORAGE_SIZE, size=BATCH)
    inputs = np.zeros((BATCH, IMG_ROWS, IMG_COLS, FRAMES_PER_SAMPLE))
    targets = np.zeros((inputs.shape[0], ACTIONS))

    # Now we do the experience replay
    for i in range(0, BATCH):
        idx = experiences[i]
        state_0, t_0, action_0, reward_0 = get_state_data(idx)
        Q_sa_0 = model.predict(state_0)
        inputs[i:i + 1] = state_0
        targets[i] = Q_sa_0
        reward = reward_0

        # Calculate the discounted reward for the given sequence depth
        for j in range(1,DISC_REWARD_DEPTH+1):
            state_j, terminal_j, action_j, reward_j = get_state_data((idx+j) % STORAGE_SIZE)

            if terminal_j:
                # The reward is not going to increase, game is over
                break
            else:
                # Predict the expected reward increase
                # We use the current model to predict the maximum future reward
                Q_sa_j = model.predict(state_j)
                # Then apply the discounted reward rule
                reward_j_max = np.max([np.max(Q_sa_j), reward_j])  # Maximum expected or experienced reward
                reward += (GAMMA**(j**GAMMA_DECAY_MUL)) * reward_j_max

        targets[i, action_0] = reward

    return model.train_on_batch(inputs, targets)


def exercise_network(model):
    global MEMORY_SIZE, MODEL_VERSION, STATE, FRAME_STORAGE, STORAGE_SIZE, STATE_POINTER, EPSILON, GAMMA, T

    T = 0               # T is the current iteration,
    c = 0               # c is a counter for the observation period
    gameover_flag = True
    last_version_change = datetime.now()

    if PROG_MODE == 0:  # Training
        # Store the previous observations in replay memory
        must_load = False
        EPSILON = INITIAL_EPSILON
        GAMMA = INITIAL_GAMMA
        print("Exercise mode: Training")

        # Memory allocation
        # The first dimension depth is 1 for keras
        global FRAME_STORAGE, REWARD_LOG, ACTIONS_LOG, TFLAG_LOG
        FRAME_STORAGE = np.zeros((1, IMG_ROWS, IMG_COLS, MEMORY_SIZE + FRAMES_PER_SAMPLE - 1), dtype=np.ubyte)
        REWARD_LOG = np.zeros(MEMORY_SIZE, dtype=np.float32)
        ACTIONS_LOG = np.zeros(MEMORY_SIZE, dtype=np.ubyte)
        TFLAG_LOG = np.zeros(MEMORY_SIZE, dtype=np.ubyte)
        TFLAG_LOG[STATE_POINTER] = 0
        print("Maximum total memory usage of experience storage =", \
              sys.getsizeof(FRAME_STORAGE) + sys.getsizeof(REWARD_LOG) + sys.getsizeof(TFLAG_LOG)), "bytes"

    else:  # Testing
        must_load = True
        EPSILON = FINAL_EPSILON
        MEMORY_SIZE = 1
        print("Exercise mode: Running")

        # We only need to store one state for testing
        FRAME_STORAGE = np.zeros((1, IMG_ROWS, IMG_COLS, FRAMES_PER_SAMPLE), dtype=np.ubyte)

    # The first state is doing nothing
    # Pre-process the image to a IMG_ROWSxIMG_COLS grayscale and repeat FRAMES_PER_SAMPLE times for the first state
    x_current, r_0, t_0, tact = ttt.step(0)

    x_current = skimage.color.rgb2gray(x_current)
    x_current = skimage.transform.resize(x_current,(IMG_ROWS,IMG_COLS))
    x_current = skimage.exposure.rescale_intensity(x_current,out_range=(0,255)).astype(np.ubyte)
    x_current = x_current.reshape(1, IMG_ROWS, IMG_COLS, 1)

    for i in range(FRAMES_PER_SAMPLE):
        FRAME_STORAGE[:, :, :, i:i+1] = x_current

    # Look for a saved state
    if RESTORE_STATE:
        lookup_state()

    # Look for a memory dump
    if RESTORE_MEMORY and PROG_MODE == 0:
        lookup_experience()
        c = STORAGE_SIZE  # Skip observation if we have enough samples

    path = SAVE_PATH()
    if os.path.exists(path):
        load_model(model, path)
    else:
        if must_load:
            print("File not found: ", path)
            exit(-1)
        else:
            print("Starting a fresh model...")

    while True:
        action_index = 0
        max_q = 0
        loss = 0

        # Update state
        if PROG_MODE == 1 or PROG_MODE == 2:
            STATE = 0
        elif c <= OBSERVE_PERIOD:
            c += 1
            STATE = 1
        elif OBSERVE_PERIOD < T <= OBSERVE_PERIOD + EXPLORE_PERIOD:
            STATE = 2
        else:
            STATE = 3

        # Choose an action
        if T % FRAMES_PER_ACTION == 0:
            q = model.predict(get_state(STATE_POINTER))  # Input a stack of FRAMES_PER_SAMPLE images, get the prediction
            max_q = np.argmax(q)
            # EPSILON is for exploration over exploitation
            if random.random() <= EPSILON:
                action_index = random.randrange(ACTIONS)
            else:
                action_index = max_q

        reward, terminal, action_index = do_action(action_index)

        if PROG_MODE == 0:  # Training

            # Version change
            if (datetime.now() - last_version_change).seconds > VERSION_UPDATE:
                MODEL_VERSION += 1
                last_version_change = datetime.now()

            # Only train if done observing, this is because our network needs a lot of data to not over fit or overflow
            if STATE > 1:

                # Reduced EPSILON gradually
                # The closer we get to a pro player, the less we should explore random strategies
                # We will reach the final value after EXPLORE_PERIOD iterations
                if EPSILON > FINAL_EPSILON:
                    EPSILON += (FINAL_EPSILON - INITIAL_EPSILON) / EXPLORE_PERIOD

                # Increase GAMMA gradually
                # The closer we get to a pro player, the more confident we should be with our predictions
                # We will reach the final value after EXPLORE_PERIOD iterations
                if GAMMA < FINAL_GAMMA:
                    GAMMA += (FINAL_GAMMA - INITIAL_GAMMA) / EXPLORE_PERIOD

                if T % THROTTLING_PERIOD == 0:
                    loss = one_iteration(model)

                # Save progress
                if T % SAVE_INTERVAL == 0:
                    save_model(model)
                    
                # Dump the experience memory
                if T % DUMPING_INTERVAL == 0:
                    dump_memory()

        if VERBOSE or T % REPORT_INTERVAL == 0 or (SIGRW and np.abs(reward) >= 0.1):
            # Print info
            if PROG_MODE == 0:
                print("V", MODEL_VERSION, "T", T, "| ST", STATE_LABELS[STATE],
                      "| ACT (BEST)", AMAP[action_index], "(",AMAP[max_q], ")| RW", reward, "| TR", terminal,
                      "| EXP", q[0][action_index],"(", q[0][max_q],
                      ") | Loss ", loss, "| EPSILON", EPSILON, "| GAMMA", GAMMA)
            elif PROG_MODE == 1 or PROG_MODE == 2:
                print("V", MODEL_VERSION, "T", T, "| S", "| ST", STATE_LABELS[STATE],
                      "| ACT (BEST)", AMAP[action_index], "(",AMAP[max_q], ") | RW", reward, "| TR", terminal,
                      "| EXP", q[0][action_index], "(", q[0][max_q], ") | EPSILON", EPSILON)

        T += 1

        if PROG_MODE == 2:
            print("Press any key for the next iteration...")
            input()

def start_work():
    model = build_model()
    exercise_network(model)


def main():
    parser = argparse.ArgumentParser(description='Train an AI to play TicTacToe')
    parser.add_argument('-m', '--mode', help='Train / Test', required=True)
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-sig', action='store_true')  # Display significant rewards
    parser.add_argument('-iw', action='store_true')  # Allowing and accounting for human intervention, enriches the meaningful data
    args = vars(parser.parse_args())
    global VERBOSE, INTERVENTION_WATCH, PROG_MODE, SIGRW
    VERBOSE = args['v']
    SIGRW = args['sig']
    INTERVENTION_WATCH = args['iw']
    if args['mode'] == 'Train':
        PROG_MODE = 0
    elif args['mode'] == 'Test':
        PROG_MODE = 1
    elif args['mode'] == 'Step':
        PROG_MODE = 2
    start_work()


if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
