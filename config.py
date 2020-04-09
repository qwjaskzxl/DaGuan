import argparse


def get_args():
    parser = argparse.ArgumentParser("TrainModel")
    # text
    parser.add_argument("--max_seq_len", type=int, default=2048*3)
    parser.add_argument("--vocab_size", type=int, default=162095)  # 1271460)
    parser.add_argument("--emb_dim", type=int, default=300)

    # model
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)

    # others
    parser.add_argument("--gpu_id", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)

    args = parser.parse_args()
    return args
