import argparse
import numpy as np
import pandas as pd
import torch
import os
import dgl


def process_friendster(dataset_path, save_path):

    def _download(url, path, filename):
        import requests

        fn = os.path.join(path, filename)
        if os.path.exists(fn):
            return
        print("Download friendster.")
        os.makedirs(path, exist_ok=True)
        f_remote = requests.get(url, stream=True)
        sz = f_remote.headers.get('content-length')
        assert f_remote.status_code == 200, 'fail to open {}'.format(url)
        with open(fn, 'wb') as writer:
            for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
                writer.write(chunk)
        print('Download finished.')

    _download(
        'https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/friendster/com-friendster.ungraph.txt.gz',
        dataset_path, 'com-friendster.ungraph.txt.gz')
    df = pd.read_csv(os.path.join(dataset_path,
                                  'com-friendster.ungraph.txt.gz'),
                     sep='\t',
                     skiprows=4,
                     header=None,
                     names=['src', 'dst'],
                     compression='gzip')
    src = df['src'].values
    dst = df['dst'].values
    print('construct the graph...')
    g = dgl.graph((src, dst))
    g = g.formats("csc")
    g.create_formats_()

    train_idx = torch.randperm(g.num_nodes())[:int(g.num_nodes() * 0.05)]

    print("Save data...")
    dgl.save_graphs(os.path.join(save_path, "graph.dgl"), [g])
    torch.save(train_idx.long(), os.path.join(save_path, "train_idx.pt"))


def process_livejournal(dataset_path, save_path):

    def _download(url, path, filename):
        import requests

        fn = os.path.join(path, filename)
        if os.path.exists(fn):
            return
        print("Download livejournal.")
        os.makedirs(path, exist_ok=True)
        f_remote = requests.get(url, stream=True)
        sz = f_remote.headers.get('content-length')
        assert f_remote.status_code == 200, 'fail to open {}'.format(url)
        with open(fn, 'wb') as writer:
            for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
                writer.write(chunk)
        print('Download finished.')

    _download(
        'https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/livejournal/soc-LiveJournal1.txt.gz',
        dataset_path, 'soc-LiveJournal1.txt.gz')
    df = pd.read_csv(os.path.join(dataset_path, 'soc-LiveJournal1.txt.gz'),
                     sep='\t',
                     skiprows=4,
                     header=None,
                     names=['src', 'dst'],
                     compression='gzip')
    src = df['src'].values
    dst = df['dst'].values
    print('construct the graph...')
    g = dgl.graph((src, dst))
    g = g.formats("csc")
    g.create_formats_()

    train_idx = torch.randperm(g.num_nodes())[:int(g.num_nodes() * 0.05)]

    print("Save data...")
    dgl.save_graphs(os.path.join(save_path, "graph.dgl"), [g])
    torch.save(train_idx.long(), os.path.join(save_path, "train_idx.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="friendster",
                        choices=["friendster", "livejournal"])
    parser.add_argument("--root", help="Path of the dataset.")
    parser.add_argument("--save-path", help="Path to save the processed data.")
    args = parser.parse_args()
    print(args)

    if args.dataset == "friendster":
        process_friendster(args.root, args.save_path)
    elif args.dataset == "livejournal":
        process_livejournal(args.root, args.save_path)
