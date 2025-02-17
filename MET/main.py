import os
import gc
import random
import pprint
import argparse
import numpy as np
from model import *
import tensorflow as tf
from utils import split_types
from framework import ZeroShotMNetFramework
from sentence_encoder import BERTSentenceEncoder
from data_loader import get_loader, MNetDataset, TextProcessor
from multimodal_encoder import MultimodalEncoder, SimpleMultimodalEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="../dataset/MET")
    parser.add_argument(
        "--bert_path", type=str, default="../pretrain/cased_L-12_H-768_A-12"
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_expert", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--model_path", type=str, default="./weights")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    task = args.task
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len

    model_path = f"{args.model_path}/model-task{task}-{args.seed}.h5"

    dataset = MNetDataset(args.data_path)
    textProcessor = TextProcessor(args.bert_path, max_seq_len)

    if task == 1:
        GroupA = ["People", "Site", "Building", "Currency"]
        GroupB = ["Location", "Event", "Book", "Music"]
        GroupC = ["Organization", "Country", "APP", "Movie"]
    elif task == 2:
        GroupC = ["People", "Site", "Building", "Currency"]
        GroupA = ["Location", "Event", "Book", "Music"]
        GroupB = ["Organization", "Country", "APP", "Movie"]
    elif task == 3:
        GroupB = ["People", "Site", "Building", "Currency"]
        GroupC = ["Location", "Event", "Book", "Music"]
        GroupA = ["Organization", "Country", "APP", "Movie"]

    train_types = GroupA
    val_types = GroupB
    test_types = GroupC

    print("train:", train_types)
    print("valid:", val_types)
    print("test:", test_types)

    train_dataloader, train_iter = get_loader(
        train_types, dataset, textProcessor, batch_size, balanced_sampling=True
    )
    val_dataloader, val_iter = get_loader(val_types, dataset, textProcessor, batch_size)
    test_dataloader, test_iter = get_loader(
        test_types, dataset, textProcessor, batch_size=1, shuffle=False
    )

    del dataset
    gc.collect()

    sentence_encoder = BERTSentenceEncoder(args.bert_path)
    multimodal_encoder = MultimodalEncoder(sentence_encoder, n_expert=args.n_expert)
    model = MMProto(multimodal_encoder, use_img=True)

    framework = ZeroShotMNetFramework(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_n_class=len(train_types),
        val_n_class=len(val_types),
        test_n_class=len(test_types),
        beta=args.beta,
    )

    if args.mode == "train":
        framework.train(
            model,
            args.lr,
            args.epoch,
            args.patience,
            train_iter,
            val_iter,
            model_path,
        )
    elif args.mode == "test":
        test_metrics, _ = framework.eval(
            model, test_iter, model_path
        )
        pprint.pprint(test_metrics)


if __name__ == "__main__":
    main()
