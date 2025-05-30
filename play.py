import argparse
import os
import time
import torch
from termcolor import colored
import colorama
colorama.init()
# local import
from src import DotsAndBoxesPrinter, AlphaBetaPlayer, AIPlayer, RandomPlayer, Checkpoint
from src.model.dual_res import AZDualRes
from src.model.feed_forward import AZFeedForward
from src.players.neural_network import NeuralNetworkPlayer
import numpy as np
from src.game import DotsAndBoxesGame

# 定义混合策略类（可直接复制 ai_vs_ai.py 中的实现）
class MixedPlayer(AIPlayer):
    def __init__(self, alphazero: NeuralNetworkPlayer, alphabeta4: AlphaBetaPlayer, alphabeta5: AlphaBetaPlayer, alphabeta6: AlphaBetaPlayer):
        super().__init__("MixedPlayer")
        self.alphazero = alphazero
        self.alphabeta4 = alphabeta4
        self.alphabeta5 = alphabeta5
        self.alphabeta6 = alphabeta6

    def determine_move(self, game: DotsAndBoxesGame) -> int:
        moves_played = np.count_nonzero(game.l != 0)
        total_moves = game.SIZE * (game.SIZE - 1) * 2
        quarter_moves = total_moves // 4

        if moves_played <= quarter_moves:
            return self.alphazero.determine_move(game)
        elif moves_played <= 2 * quarter_moves:
            return self.alphabeta4.determine_move(game)
        elif moves_played <= 3 * quarter_moves:
            return self.alphabeta5.determine_move(game)
        else:
            return self.alphabeta6.determine_move(game)

class DepMixedPlayer(AIPlayer):
    def __init__(self, alphabeta3: AlphaBetaPlayer, alphabeta4: AlphaBetaPlayer, alphabeta5: AlphaBetaPlayer):
        super().__init__("DepMixedPlayer")
        self.alphabeta3 = alphabeta3
        self.alphabeta4 = alphabeta4
        self.alphabeta5 = alphabeta5

    def determine_move(self, game: DotsAndBoxesGame) -> int:
        moves_played = np.count_nonzero(game.l != 0)
        total_moves = game.SIZE * (game.SIZE - 1) * 2
        third_moves = total_moves // 3

        if moves_played <= third_moves:
            return self.alphabeta3.determine_move(game)
        elif moves_played <= 2 * third_moves:
            return self.alphabeta4.determine_move(game)
        else:
            return self.alphabeta5.determine_move(game)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=2,
                    help='Size of the Dots-and-Boxes game (in number of boxes per row and column).')
parser.add_argument('-o', '--opponent', type=str, default="alpha_beta",
                    choices=["person", "random", "alpha_beta", "alpha_zero", "mixed", "depmix"],
                    help='Type of opponent to play against.')
parser.add_argument('-cp', '--checkpoint', type=str,
                    help='In case of AlphaZero as opponent: Model checkpoint (i.e., name of folder containing config and model).')
parser.add_argument('-d', '--depth', type=int, default=3,
                    help='In case of Alpha–beta pruning as opponent: Specifies the search depth')
parser.add_argument('-f', '--first', type=int, default=1,
                    help='Fisrt==1 The AI plays First, First==0 The human player plays first. Default: 1')
args = parser.parse_args()


def cls(): os.system("cls" if os.name == "nt" else "clear")


def main(size: int, opponent: AIPlayer, AI_First: int):
    cls()

    opponent_name = "Opponent" if opponent is None else opponent.name
    game = DotsAndBoxesPrinter(size, opponent_name)

    print(game.state_string())
    print(game.board_string())

    if(AI_First == 1):
        game.current_player = -1
    else:
        game.current_player = 1
    while game.is_ensuring():

        if game.current_player == 1 or opponent is None:
            # print draw request
            # process draw request
            while True:
                move = input("Please enter a free line number: ")
                if move.isdigit() == False and move != "-1":
                    print("Line number must be a number.")
                elif int(move) in game.get_valid_moves() or int(move) == -1:#-1 is used to rollback the game
                    move = int(move)
                    break
                print(f"Line {move} is not a valid move. Please select a move in {game.get_valid_moves()}.")
            
            last_move_by_player = True

        else:
            # an AI opponent is at turn
            #time.sleep(1.0)
            start_time = time.time()
            game.push_history()
            move = opponent.determine_move(game)
            stopped_time = time.time() - start_time
            last_move_by_player = False
        if move == -1 :
            # rollback the game
            if game.rollback():
                #cls()
                print("\nRollback the game successful")                
                print(game.state_string())
                print(game.board_string())
            else:
                print("Rollback the game failed: No moves to rollback")
        else:
            
            game.push_history()
            game.execute_move(move)

            # print new game state
            #cls()
            if not last_move_by_player:
                print("Computation time of opponent for previous move " + colored("{0:.2f}s".format(stopped_time),"green"))
                print( "AI move position:" + colored(f" {move}", "green") )
            else:
                print()
            print(game.state_string())
            print(game.board_string())
        if(game.result is not None):
            result_ensuring = input(f"Do you want to ensure the result? (y/n): {'You wins' if game.result == 1 else ('AI wins' if game.result == -1 else 'Draw')} :")
            if(result_ensuring.lower() == "y"):
                game.set_running(False)
            else:
                #cls()
                game.rollback()
                game.set_running(None)
                game.set_result(None)
                print("\nRollback the game successful")                
                print(game.state_string())
                print(game.board_string())
    

    if game.result == 1:
        print("The game is over.. You won!")
    elif game.result == -1:
        print("The game is over.. You lost :(")
    else:
        print("The game ended in a draw ..")
    print(game.state_string())



if __name__ == '__main__':

    game_size = args.size

    if args.opponent == "person":
        opponent = None

    elif args.opponent == "random":
        opponent = RandomPlayer()

    elif args.opponent == "alpha_beta":
        opponent = AlphaBetaPlayer(depth=args.depth)

    elif args.opponent == "alpha_zero":
        LOGS_FOLDER = "logs/"

        checkpoint_folder = LOGS_FOLDER + args.checkpoint + "/"
        if not os.path.exists(checkpoint_folder):
            exit(f"loading Checkpoint failed: {checkpoint_folder} does not exist")

        # create checkpoint handler and load config
        checkpoint = Checkpoint(checkpoint_folder)
        config = checkpoint.load_config()
        inference_device = "cpu"
        assert config["game_size"] == args.size

        # initialize model
        AZModel = None
        if config["model_parameters"]["name"] == "FeedForward":
            AZModel = AZFeedForward
        elif config["model_parameters"]["name"] == "DualRes":
            AZModel = AZDualRes

        model = AZModel(
            game_size=game_size,
            inference_device="cpu",
            model_parameters=config["model_parameters"],
        ).float()
        # 直接加载完整权重，不再过滤头部
        state_dict = torch.load(checkpoint.model)
        model.load_state_dict(state_dict)

        opponent = NeuralNetworkPlayer(
            model=model,
            name=f"AlphaZero({game_size}x{game_size})",
            mcts_parameters=config["mcts_parameters"],
            device="cpu"
        )
    elif args.opponent == "mixed":
        if not args.checkpoint:
            raise ValueError("必须通过 --checkpoint 指定 AlphaZero 模型路径")
        
        # 加载 AlphaZero 模型
        checkpoint_folder = args.checkpoint + "/"
        if not os.path.exists(checkpoint_folder):
            raise FileNotFoundError(f"Checkpoint 路径不存在: {checkpoint_folder}")
        
        checkpoint = Checkpoint(checkpoint_folder)
        config = checkpoint.load_config()
        model = AZDualRes(game_size=game_size, inference_device="cpu", model_parameters=config["model_parameters"])
        model.load_checkpoint(checkpoint.model)
        
        alphazero = NeuralNetworkPlayer(
            model=model,
            name="AlphaZero",
            mcts_parameters=config["mcts_parameters"],
            device="cpu"
        )
        
        # 创建不同深度的 AlphaBetaPlayer
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=6)
        alphabeta6 = AlphaBetaPlayer(depth=8)
        
        opponent = MixedPlayer(alphazero, alphabeta4, alphabeta5, alphabeta6)

    elif args.opponent == "depmix":
        # 创建不同深度的 AlphaBetaPlayer
        alphabeta3 = AlphaBetaPlayer(depth=3)
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=5)
        opponent = DepMixedPlayer(alphabeta3, alphabeta4, alphabeta5)
    main(game_size, opponent, args.first)
'''(完整路径checkpoint)
# 使用 MixedPlayer 作为对手
python play.py --size 5 --opponent mixed --checkpoint D:\WORK2\回档\base\AlphaZero-for-Dots-and-Boxes-Transfer-Learning\logs\transfer_5x5 

# 使用 DepMixedPlayer 作为对手
python play.py --size 5 --opponent depmix 
'''