from argparse import ArgumentParser
from src.models.prefix_tuning import PrefixBart
from src.training.data_loader import DataModule

import pytorch_lightning as pl

if __name__ == 'main':
    parser = ArgumentParser()
    parser.add_argument("--dataset_name_and_path", default="rcmrc", type=str)
    parser.add_argument("--seed", default=9527, type=int,
                        help="random seed for initialization")
    parser.add_argument("--model_name_and_path", default=None, type=str,
                        help="path to the pretrained")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for test and val")
    parser.add_argument("--output_dir", default='./', type=str,
                        help="The output data dir")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to run eval on the test set")

    parser = PrefixBart.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data_module = DataModule(**vars(args))
    model = PrefixBart(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)
