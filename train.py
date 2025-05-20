import os
import time
import yaml
import argparse

# local import
from trainer import Trainer
from src import Checkpoint


RESOURCES_FOLDER = "resources/"
LOGS_FOLDER = "logs/"


parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--checkpoint', type=str, default=None,
                    help='Model checkpoint (i.e., name of folder which saved data) to start training from.')
parser.add_argument('-cf', '--config', type=str, required=True,
                    help='name of config which should be used for training (only relevant for new training)')
parser.add_argument('-w', '--n_workers', type=int, default=1,
                    help='Number of threads during self-play. Each thread performs games of self-play.')
parser.add_argument('-idev', '--inference_device', type=str, default="cpu", choices=["cpu", "cuda"],
                    help='Device with which model interference is performed during MCTS.')
parser.add_argument('-tdev', '--training_device', type=str, default="cuda", choices=["cpu", "cuda"],
                    help='Device with which model training is performed.')
parser.add_argument('-tf', '--transfer_from', type=int, default=0,
                    help='If specified, load weights from a pre-trained model of this size (e.g., 4 for 4x4 model). Default: 0 (no transfer).')

args = parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)

    if not args.checkpoint:
        # new training
        checkpoint_folder = LOGS_FOLDER + time.strftime("%d%m%Y-%H%M%S") + "/"

        # create checkpoint handler
        checkpoint = Checkpoint(checkpoint_folder)

        # load config
        config_file = RESOURCES_FOLDER + args.config + ".yaml"
        with open(config_file) as f:
            config = yaml.safe_load(f)

    else:
        # continue training from checkpoint, load config file
        checkpoint_folder = LOGS_FOLDER + args.checkpoint + "/"
        if not os.path.exists(checkpoint_folder):
            exit(f"loading Checkpoint failed: {checkpoint_folder} does not exist")
        print(f"training is continued with data in {checkpoint_folder}")

        # create checkpoint handler
        checkpoint = Checkpoint(checkpoint_folder)

        # load config from checkpoint
        config = checkpoint.load_config()

        # 如果是迁移训练，加载预训练模型配置
        if args.config.endswith('_transfer'):
            pretrained_config_file = RESOURCES_FOLDER + args.config + ".yaml"
            with open(pretrained_config_file) as f:
                pretrained_config = yaml.safe_load(f)
            config['model_parameters'] = pretrained_config['model_parameters']

    # 新增迁移训练逻辑
    pretrained_weights = None
    if args.transfer_from > 0:
        transfer_model_folder = LOGS_FOLDER + f"alpha_zero_{args.transfer_from}x{args.transfer_from}" + "/"
        if not os.path.exists(transfer_model_folder):
            exit(f"Loading pre-trained weights failed: {transfer_model_folder} does not exist")
        print(f"Loading pre-trained weights from {transfer_model_folder}")
        # 创建临时CheckPoint
        transfer_checkpoint = Checkpoint(transfer_model_folder)
        # 加载预训练模型的配置
        pretrained_config = transfer_checkpoint.load_config()
        # 确保模型结构一致
        config['model_parameters'] = pretrained_config['model_parameters']
        # 加载预训练权重
        pretrained_weights = transfer_checkpoint.model

    trainer = Trainer(
        config=config,
        n_workers=args.n_workers,
        inference_device=args.inference_device,
        training_device=args.training_device,
        checkpoint=checkpoint,
        pretrained_weights=pretrained_weights  # 传递预训练权重
    )
    trainer.loop()


