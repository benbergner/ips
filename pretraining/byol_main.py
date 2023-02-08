#-*- coding:utf-8 -*-
import os
#working_dir = "/dhc/home/benjamin.bergner/netstore-old/projects/gigapixel/giganet"
#os.chdir(working_dir)
#sys.path.append(working_dir)

from pathlib import Path
import yaml
import torch
import torch.distributed as dist

from pretraining.byol_trainer import BYOLTrainer
from pretraining.utils import logging_util

def run_task(config):
    logging = logging_util.get_std_logging()
    if config['distributed']:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        config.update({'world_size': world_size, 'rank': rank, 'local_rank': local_rank})

        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        logging.info(f'world_size {world_size}, gpu {local_rank}, rank {rank} init done.')
    else:
        config.update({'world_size': 1, 'rank': 0, 'local_rank': 0})

    trainer = BYOLTrainer(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    for epoch in range(start_epoch + 1, trainer.total_epochs + 1):
        trainer.train_epoch(epoch, printer=logging.info)
        trainer.save_checkpoint(epoch)

def main():
    with open(Path(Path(__file__).parent, 'train_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    run_task(config)

if __name__ == "__main__":
    print(f'Pytorch version: {torch.__version__}')
    print(f'os environ: {os.environ}')
    main()