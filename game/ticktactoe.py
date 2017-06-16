from abc import abstractmethod

import pygame
from pygame.locals import *
import random

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)
SYMBOLS = ["O", "X"]
E = -1
X = 1
O = 0

# AI Types
MIN_MAX = 0
DQN = 1

class TTTGame:
    difficulty = 1  # Correct move probability
    waiting = False

    def __init__(self, width, height, ai_type):
        self.width = width
        self.height = height
        self.h_cell = width / 3
        self.v_cell = height / 3
        self.clock = pygame.time.Clock()
        pygame.init()
        self.window = pygame.display.set_mode((width, height))
        self.screen = pygame.display.get_surface()
        pygame.display.set_caption("TicTacToe")
        font = pygame.font.SysFont("arial", 32)
        self.x_g = font.render(SYMBOLS[X], 1, WHITE)
        self.o_g = font.render(SYMBOLS[O], 1, WHITE)
        self.x_g_hg = font.render(SYMBOLS[X], 1, GREEN)
        self.o_g_hg = font.render(SYMBOLS[O], 1, GREEN)
        self.x_g_hr = font.render(SYMBOLS[X], 1, RED)
        self.o_g_hr = font.render(SYMBOLS[O], 1, RED)
        x_sz = self.x_g.get_size()
        o_sz = self.o_g.get_size()
        self.x_xd, self.x_yd = (x_sz[0] / 2, x_sz[1] / 2)
        self.o_xd, self.o_yd = (o_sz[0] / 2, o_sz[1] / 2)
        self.load_ai(ai_type)
        self.new_game()

    def load_ai(self, ai_type):
        if int(ai_type) == DQN:
            print("Using DQN AI")
            self.ai_machine = TrainedDQNAI(self)
        else:
            print("Using MinMax AI")
            self.ai_machine = MinMaxAI()

    def new_game(self):
        self.waiting = False
        self.state = [E] * 9
        self.winning_comb = [E] * 3
        self.turn = random.randint(0, 1)
        self.player = random.randint(0, 1)
        self.ai = 1 - self.player
        self.moves = 0
        self.gameover = False
        self.winner = E
        self.ai_machine.reset_state()
        self.redraw()

    def check_input(self, events):
        for event in events:
            if event.type == QUIT:
                exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.waiting:
                    self.new_game()
                elif self.turn == self.player:
                    mp = pygame.mouse.get_pos()
                    mx = int(mp[0] / self.h_cell)
                    my = int(mp[1] / self.v_cell)
                    idx = mx + my * 3
                    if self.state[idx] == E:
                        self.player_move(idx)
                        return True

        return False  # No changes

    def highlight_win(self):
        for i in range(0, 3):
            idx = self.winning_comb[i]
            x = (idx % 3) * self.h_cell + self.h_cell / 2
            y = int(idx / 3) * self.v_cell + self.v_cell / 2
            if self.winner == X:
                if self.winner == self.player:
                    self.screen.blit(self.x_g_hg, (x - self.x_xd, y - self.x_yd))
                else:
                    self.screen.blit(self.x_g_hr, (x - self.x_xd, y - self.x_yd))
            else:
                if self.winner == self.player:
                    self.screen.blit(self.o_g_hg, (x - self.o_xd, y - self.o_yd))
                else:
                    self.screen.blit(self.o_g_hr, (x - self.o_xd, y - self.o_yd))
        pygame.display.update()

    def draw(self):
        for i in range(1, 4):
            x = i * self.h_cell
            pygame.draw.line(self.screen, WHITE, (x, 0), (x, self.height))
        for i in range(1, 4):
            y = i * self.v_cell
            pygame.draw.line(self.screen, WHITE, (0, y), (self.width, y))
        for i in range(0, 9):
            x = (i % 3) * self.h_cell + self.h_cell / 2
            y = int(i / 3) * self.v_cell + self.v_cell / 2
            if self.state[i] == X:
                self.screen.blit(self.x_g, (x - self.x_xd, y - self.x_yd))
            elif self.state[i] == O:
                self.screen.blit(self.o_g, (x - self.o_xd, y - self.o_yd))

    def redraw(self):
        self.screen.fill(BLACK)
        self.draw()
        pygame.display.update()

    def check_game(self):
        for i in range(0, 3):
            # Check vertical
            if self.state[i] == self.state[i + 3] == self.state[i + 6] != E:
                self.gameover = True
                self.winning_comb = [i, i + 3, i + 6]
                self.winner = self.state[i]
                return
            # Check horizontal
            j = i * 3
            if self.state[j] == self.state[j + 1] == self.state[j + 2] != E:
                self.gameover = True
                self.winning_comb = [j, j + 1, j + 2]
                self.winner = self.state[j]
                return
        # Check diagonals
        if self.state[0] == self.state[4] == self.state[8] != E:
            self.gameover = True
            self.winning_comb = [0, 4, 8]
            self.winner = self.state[0]
            return
        if self.state[2] == self.state[4] == self.state[6] != E:
            self.gameover = True
            self.winning_comb = [2, 4, 6]
            self.winner = self.state[2]
            return

        if self.moves == 9:
            self.gameover = True

    def ai_move(self, dif_override=-1):
        if dif_override > -1:
            dif = dif_override
        else:
            dif = self.difficulty
        move = self.ai_machine.decide_move(dif)
        self.state[move] = self.ai
        self.turn = self.player
        self.moves += 1
        self.check_game()

    def player_move(self, idx):
        self.ai_machine.external_move(idx)
        self.state[idx] = self.player
        self.turn = self.ai
        self.moves += 1
        self.check_game()

    def snapshot(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def step(self, fps):
        if self.turn == self.ai:
            self.ai_move()
            self.redraw()
        elif self.check_input(pygame.event.get()):
            self.redraw()
        self.check_win()
        self.clock.tick(fps)

    def check_win(self):
        if self.gameover:
            self.gameover = False
            if self.winner == self.player:
                txt = SYMBOLS[self.winner] + " - Player Won"
                self.highlight_win()
            elif self.winner == self.ai:
                txt = SYMBOLS[self.winner] + " - AI Won"
                self.highlight_win()
            else:
                txt = "The game ended in a tie"
            print(txt)
            self.waiting = True
            self.turn = self.player

    def tick(self, fps):
        pygame.event.get()
        self.clock.tick(fps)


class AIMachine:
    def reset_state(self):
        ...

    def external_move(self, idx):
        ...

    @abstractmethod
    def decide_move(self):
        ...


class MinMaxAI(AIMachine):
    __metaclass_ = AIMachine
    def __init__(self):
        self.state = 0               # The state represents 0s (Lower 9 bits) and 1s (Higher 9 bits)
        self.turn = 0                # Current turn, used to decide on next move
        self.memory = [-1] * 245791  # Stores the min/max value (Upper nibble) and the associated move (Lower nibble)
        self.calc_minmax(0, 0)       # Generate the decision memory

    # Player 0 is min (Win=0)
    # Player 1 is max (Win=2)
    def calc_minmax(self, state, player):
        # The same state cannot occur for different players so we can apply dp
        if self.memory[state] > -1:
            return (self.memory[state] & 0xF0) >> 4
        # Base case, winning or a tie
        for i in range(0, 3):
            # Check vertical
            if (state & (1 << i)) and (state & (1 << (i + 3)) and (state & (1 << i + 6))):
                self.memory[state] = 0
                return 0
            if (state & (1 << i + 9)) and (state & (1 << (i + 12)) and (state & (1 << i + 15))):
                self.memory[state] = 2 << 4
                return 2
            # Check horizontal
            j = i * 3
            if (state & (1 << j)) and (state & (1 << (j + 1))) and (state & (1 << (j + 2))):
                self.memory[state] = 0
                return 0
            if (state & (1 << (j + 9))) and (state & (1 << (j + 10))) and (state & (1 << (j + 11))):
                self.memory[state] = 2 << 4
                return 2
        # Check diagonals
        if (state & 1) and (state & (1 << 4)) and (state & (1 << 8)):
            self.memory[state] = 0
            return 0
        if (state & 1 << 9) and (state & (1 << 13)) and (state & (1 << 17)):
            self.memory[state] = 2 << 4
            return 2
        if (state & (1 << 2)) and (state & (1 << 4)) and (state & (1 << 6)):
            self.memory[state] = 0
            return 0
        if (state & 1 << 11) and (state & (1 << 13)) and (state & (1 << 15)):
            self.memory[state] = 2 << 4
            return 2
        if ((state & 0x1FF) | ((state & 0x3FE00) >> 9)) == 0x1FF:  # Tie
            self.memory[state] = 1 << 4
            return 1

        # Traverse all possible states for the player and calculate the min/max according to his type
        if player == 0:
            val = 3
        else:
            val = -1
        idx = 0
        for i in range(0, 9):
            if not (state & (1 << i) or state & (1 << (i + 9))):
                mask_idx = 9 * player + i
                val_i = self.calc_minmax(state | (1 << mask_idx), 1 - player)  # Calculate for doing this action
                if player == 0:  # Min case
                    if val_i < val:
                        val = val_i
                        idx = i
                elif val_i > val:  # Max case
                    val = val_i
                    idx = i

        # Set the memory value
        self.memory[state] = idx | (val << 4)

        # Return min/max for the recursive call
        return val

    def reset_state(self):
        self.state = 0
        self.turn = 0

    def external_move(self, idx):
        self.state = self.state | (1 << (9 * self.turn + idx))
        self.turn = 1 - self.turn

    def decide_move(self, dif):
        # Choose whether to play best move or random move based on difficulty
        if random.random() < dif:
            move = self.memory[self.state] & 0xF
        else:
            possible_moves = []
            for i in range(1, 9):
                if not (self.state & (1 << i) or self.state & (1 << (i + 9))):
                    possible_moves.append(i)
            move = possible_moves[random.randint(0, len(possible_moves)-1)]
        self.state = self.state | (1 << (9 * self.turn + move))
        self.turn = 1 - self.turn
        return move


class TrainedDQNAI(AIMachine):
    __metaclass_ = AIMachine
    IMG_ROWS = 60
    IMG_COLS = 60
    path = ".\\Jon\\latest"

    def __init__(self, game):
        import sys
        if sys.path[0].endswith('game'):
            self.path = "."+self.path
        self.model = self.create_ai()
        self.game = game

    def create_ai(self):
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation, Flatten
        from keras.layers.convolutional import Convolution2D, MaxPooling2D
        from keras.optimizers import Adam
        model = Sequential()
        cell_sz = (int(self.IMG_ROWS / 3), int(self.IMG_COLS / 3))
        model.add(Convolution2D(3, kernel_size=cell_sz, strides=cell_sz, padding='same',
                                input_shape=(self.IMG_ROWS, self.IMG_COLS, 1)))
        model.add(Activation('relu'))
        model.add(Convolution2D(81, kernel_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(9))
        model.load_weights(self.path)
        adam = Adam()
        model.compile(loss='mse', optimizer=adam)
        return model
    
    def decide_move(self, dif):
        if random.random() < dif:
            import skimage
            from skimage import transform, color, exposure
            import numpy as np
            x = skimage.color.rgb2gray(self.game.snapshot())
            x = skimage.transform.resize(x, (self.IMG_ROWS, self.IMG_COLS), mode='constant')
            x = skimage.exposure.rescale_intensity(x, out_range=(0, 255)).astype(np.ubyte)
            x = x.reshape(1, self.IMG_ROWS, self.IMG_COLS, 1)
            p = self.model.predict(x)
            move = np.argmax(p)
            # print("Model prediction:", p, "Move:", move)
            if self.game.state[move] == E:
                return move
        # print("Random move")
        possible_moves = []
        for i in range(0, 9):
            if self.game.state[i] == E:
                possible_moves.append(i)
        return possible_moves[random.randint(0, len(possible_moves) - 1)]

