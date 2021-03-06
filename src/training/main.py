from argparse import ArgumentParser

import pytorch_lightning as pl

from src.training.data_loader import DataModule
from src.training.configs import MODEL_CLASSES
from src.training.trainer import generate_trainer


def main(args):
    pl.seed_everything(args.seed)

    data_module = DataModule(args)
    model = MODEL_CLASSES[args.model_type][1](args)

    trainer = generate_trainer(args)

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        result = trainer.test(model, data_module)
        print(result)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--output_dir", default=None, required=True, type=str, help="Path to save, load results")
    parser.add_argument("--data_dir", default=None, required=True, type=str, help="The input data dir")

    parser.add_argument("--cache_dir", default='./', type=str,
                        help="Path to directory in which a downloaded pretrained model and dataset")

    parser.add_argument("--model_type", default="bart", type=str, help="Model type selected")

    parser.add_argument("--seed", default=9527, type=int, help="Random seed for initialization")
    parser.add_argument("--train_batch_size", default=256, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for test and val")
    parser.add_argument("--learning_rate", default=5e-5, help="The initial learning rate for Adam")
    parser.add_argument("--pre_len", default=200, type=int, help="The prefix embedding length")
    parser.add_argument("--prefix_dropout", default=0.3, type=float, help="The dropout rate for prefix")
    parser.add_argument("--max_question_length", default=64, type=int, help="The question token max length")
    parser.add_argument("--doc_stride", default=128, type=int, help="The stride of slide window for generate the "
                                                                    "context token")

    parser.add_argument("--model_name_or_path", default=None, type=str, help="path to the pretrained")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing")

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
