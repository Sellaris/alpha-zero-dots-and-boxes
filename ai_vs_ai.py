import argparse
from src.evaluator import Evaluator
from src.players.alpha_beta import AlphaBetaPlayer
from src.players.random import RandomPlayer
from src.players.neural_network import NeuralNetworkPlayer
from src.model.dual_res import AZDualRes
from src.utils.checkpoint import Checkpoint
import numpy as np
from src.players.player import AIPlayer
from src.game import DotsAndBoxesGame
class depmixed_player(AIPlayer):
    def __init__(self,  alphabeta3: AlphaBetaPlayer, alphabeta4: AlphaBetaPlayer, alphabeta5: AlphaBetaPlayer):
        super().__init__("depmixed_player")
        self. alphabeta3 = alphabeta3
        self. alphabeta4 = alphabeta4
        self. alphabeta5 = alphabeta5
    def determine_move(self, game: DotsAndBoxesGame) -> int:
        moves_played = np.count_nonzero(game.l != 0)  # 已下总手数
        total_moves = game.SIZE * (game.SIZE - 1) *2  # 游戏最大手数
        half_moves = total_moves // 3  # 30%阈值

        if moves_played <= half_moves:
            return self.alphabeta3.determine_move(game)  # 前30%使用AlphaZero3
        elif moves_played  <= total_moves:
            return self.alphabeta4.determine_move(game)  # 后30%切换Alpha-Beta4
        else:
            return self.alphabeta5.determine_move(game)  # 后30%切换Alpha-Beta5
class MixedPlayer(AIPlayer):
    def __init__(self, alphazero: NeuralNetworkPlayer, alphabeta4: AlphaBetaPlayer,alphabeta5: AlphaBetaPlayer):
        super().__init__("MixedPlayer")
        self.alphazero = alphazero
        self.alphabeta4 = alphabeta4
        self.alphabeta5 = alphabeta5

    def determine_move(self, game: DotsAndBoxesGame) -> int:
        moves_played = np.count_nonzero(game.l != 0)  # 已下总手数
        total_moves = game.SIZE * (game.SIZE - 1) *2  # 游戏最大手数
        half_moves = total_moves // 3  # 30%阈值

        if moves_played <= half_moves:
            return self.alphazero.determine_move(game)  # 前30%使用AlphaZero
        elif moves_played <= 2* half_moves:
            return self.alphabeta4.determine_move(game)
        else:
            return self.alphabeta5.determine_move(game)  # 后30%切换Alpha-Beta
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_size', type=int, default=3)
    parser.add_argument('--n_games', type=int, default=50)
# 修改参数解析部分
    parser.add_argument('--player1', choices=['alpha_beta', 'random', 'alpha_zero', 'mixed', 'depmix'], default='alpha_beta')
    parser.add_argument('--player2', choices=['alpha_beta', 'random', 'alpha_zero', 'mixed', 'depmix'], default='random')    
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (required if player is alpha_zero)')
    args = parser.parse_args()

    # 初始化玩家
    if args.player1 == 'alpha_beta':
        player1 = AlphaBetaPlayer(depth=2)
    elif args.player1 == 'random':
        player1 = RandomPlayer()
    elif args.player1 == 'mixed':
        if not args.checkpoint:
            raise ValueError("Checkpoint必须通过--checkpoint指定以加载AlphaZero模型")
        config = Checkpoint(args.checkpoint).load_config()
        model = AZDualRes(game_size=args.game_size, inference_device="cpu", model_parameters=config["model_parameters"])
        model.load_checkpoint(args.checkpoint + "/model.pt")
        alphazero = NeuralNetworkPlayer(
            model=model,
            name="AlphaZero",
            mcts_parameters=config["mcts_parameters"],
            device="cpu"
        )
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=5)
        player1 = MixedPlayer(alphazero, alphabeta4, alphabeta5)
    elif args.player1 == 'depmix':
        alphabeta3 = AlphaBetaPlayer(depth=3)
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=5)
        player1 = depmixed_player(alphabeta3, alphabeta4, alphabeta5)
    else:
        # 需加载模型
        config = Checkpoint(args.checkpoint).load_config()
        model = AZDualRes(game_size=args.game_size, inference_device="cpu", model_parameters=config["model_parameters"])
        model.load_checkpoint(args.checkpoint + "/model.pt")
        player1 = NeuralNetworkPlayer(model=model, name="AlphaZero", mcts_parameters=config["mcts_parameters"], device="cpu")

    if args.player2 == 'alpha_beta':
        player2 = AlphaBetaPlayer(depth=args.depth)
    elif args.player2 == 'random':
        player2 = RandomPlayer()
    elif args.player2 == 'mixed':
        if not args.checkpoint:
            raise ValueError("Checkpoint必须通过--checkpoint指定以加载AlphaZero模型")
        config = Checkpoint(args.checkpoint).load_config()
        model = AZDualRes(game_size=args.game_size, inference_device="cpu", model_parameters=config["model_parameters"])
        model.load_checkpoint(args.checkpoint + "/model.pt")
        alphazero = NeuralNetworkPlayer(
            model=model,
            name="AlphaZero",
            mcts_parameters=config["mcts_parameters"],
            device="cpu"
        )
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=5)
        player1 = MixedPlayer(alphazero, alphabeta4, alphabeta5)
    elif args.player2 == 'depmix':
        alphabeta3 = AlphaBetaPlayer(depth=3)
        alphabeta4 = AlphaBetaPlayer(depth=4)
        alphabeta5 = AlphaBetaPlayer(depth=5)
        player2 = depmixed_player(alphabeta3, alphabeta4, alphabeta5)
    else:
        config = Checkpoint(args.checkpoint).load_config()
        model = AZDualRes(game_size=args.game_size, inference_device="cpu", model_parameters=config["model_parameters"])
        model.load_checkpoint(args.checkpoint + "/model.pt")
        player2 = NeuralNetworkPlayer(model=model, name="AlphaZero", mcts_parameters=config["mcts_parameters"], device="cpu")

    # 开始对战
    evaluator = Evaluator(
        game_size=args.game_size,
        player1=player1,
        player2=player2,
        n_games=args.n_games,
        n_workers=4  # 可调整线程数
    )
    evaluator.compare()

if __name__ == '__main__':
    main()
'''
# 示例1：AlphaBeta vs Random (3x3棋盘，100局)
python ai_vs_ai.py --game_size 3 --n_games 100 --player1 alpha_beta --player2 random --depth 3

# 示例2：俩个AlphaZero模型对战 (需指定checkpoint路径,请使用绝对路径)
python ai_vs_ai.py --game_size 3 --n_games 50 --player1 alpha_zero --player2 alpha_zero --checkpoint logs/alpha_zero_3x3

# 示例3：混合策略 vs AlphaBeta
python ai_vs_ai.py --game_size 5  --n_games 50 --player1 mixed  --player2 alpha_beta  --depth 3  --checkpoint D:\WORK2\回档\base\AlphaZero-for-Dots-and-Boxes-Transfer-Learning\logs\transfer_5x5           
# 示例4：混合策略 vs depmix-AlphaBeta
python ai_vs_ai.py --game_size 5  --n_games 50 --player1 mixed  --player2 depmix  --checkpoint D:\WORK2\回档\base\AlphaZero-for-Dots-and-Boxes-Transfer-Learning\logs\transfer_5x5           
'''
'''
alphazero -4 vs dep2 78%
alphazero -4 vs dep3 74%

alphazero -4 vs dep -3 -4 -5
alphazero -4 -5 vs dep -3 -4 -5
alphazero -4 -5 vs dep -3
'''