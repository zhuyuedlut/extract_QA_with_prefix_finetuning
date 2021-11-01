from argparse import ArgumentParser

import pytorch_lightning as pl

from src.training.data_loader import DataModule
from src.training.configs import MODEL_CLASSES
from src.training.trainer import generate_trainer


def main(args):
    pl.seed_everything(args.seed)

    data_module = DataModule(**vars(args))
    model = MODEL_CLASSES[args.model_type][1](**vars(args))

    trainer = generate_trainer(args)

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        trainer.test(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load models")

    parser.add_argument("--cache_dir", default='./', type=str,
                        help="Path to directory in which a downloaded pretrained model and dataset")

    parser.add_argument("--model_type", default="bart", type=str, help="Model type selected")

    parser.add_argument("--seed", default=9527, type=int, help="Random seed for initialization")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for test and val")
    parser.add_argument("--learning_rate", default=5e-5, help="The initial learning rate for Adam")
    parser.add_argument("--pre_len", default=200, type=int, help="The prefix embedding length")

    parser.add_argument("--model_name_or_path", default=None, type=str, help="path to the pretrained")

    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--val_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing")

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
