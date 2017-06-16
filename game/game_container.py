import game.ticktactoe as ttt
import argparse
import pylab as plt

mode = 2
valid_rw = 0
invalid_pn = -1
won_rw = 1
lost_pn = -0.8  # If losing penalty is >= invalid move penalty, the game will eventually chose getting stuck over losing
tie_rw = 0.8

game_count = 0
win_count = 0
lose_count = 0

WIDTH = 300
HEIGHT = 300
FPS = 100


def step(action):
    terminal = 0
    reward = valid_rw
    if game.waiting:
        game.new_game()
        print("New game: Player", ttt.SYMBOLS[game.player])
        if game.turn == game.ai:
            print("AI starting")
            game.ai_move(0)  # Random first move
        else:
            print("Player starting")
        game.redraw()
    else:
        if game.state[action] == ttt.E:
            game.player_move(action)
            if not game.gameover:
                game.ai_move()
            game.redraw()
            game.check_win()
        else:
            reward = invalid_pn
        if game.waiting:
            global game_count, win_count, lose_count
            game_count += 1
            terminal = 1
            if game.winner == game.player:
                win_count += 1
                reward = won_rw
            elif game.winner == game.ai:
                lose_count += 1
                reward = lost_pn
            else:
                reward = tie_rw
            print("Lost:", lose_count/game_count*100, "% Tie:", (game_count-win_count-lose_count)/game_count*100,
                  "% Won", win_count/game_count*100, "%")
    image = game.snapshot()
    game.tick(FPS)
    return [image, reward, terminal, action]


def main():
    parser = argparse.ArgumentParser(description='A Pacman game.')
    parser.add_argument('-run', action='store_true')
    parser.add_argument('-step', action='store_true')
    parser.add_argument('-a', '--ai', help='0=MinMax / 1=DQN')
    args = vars(parser.parse_args())
    global mode, game
    if args['run']:
        mode = 1
    if args['step']:
        mode = 2
    if args['ai'] is None:
        game = ttt.TTTGame(WIDTH, HEIGHT, 0)
    else:
        game = ttt.TTTGame(WIDTH, HEIGHT, args['ai'])

if __name__ == "__main__":
    main()
else:
    global game
    # Train the model vs an older version
    game = ttt.TTTGame(WIDTH, HEIGHT, 0)

if mode == 1:
    while True:
        game.step(FPS)