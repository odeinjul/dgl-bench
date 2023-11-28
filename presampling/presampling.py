import argparse
import time
import torch
import torch.distributed as dist
import dgl
import os

import sys

sys.path.append("utils")
from load_graph import load_ogb, load_friendster, load_livejournal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--fan_out", type=str, default="5,10,15")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help=
        "datasets: ogbn-products, ogbn-papers100M, friendster, livejournal",
    )
    parser.add_argument("--root", type=str, default="/data")
    parser.add_argument("--save_path", type=str, default=".")
    args = parser.parse_args()

    print(args)

    if args.dataset == "ogbn-products":
        g, num_classes = load_ogb("ogbn-products", args.root)
    elif args.dataset == "ogbn-papers100M":
        g, num_classes = load_ogb("ogbn-papers100M", args.root)
    elif args.dataset == "friendster":
        g, num_classes = load_friendster(args.root)
    elif args.dataset == "livejournal":
        g, num_classes = load_livejournal(args.root)

    dist.init_process_group(init_method='tcp://127.0.0.1:12347',
                            rank=0,
                            world_size=1)

    train_nid = g.ndata["train_mask"].nonzero().flatten()
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")])
    dataloader = dgl.dataloading.DataLoader(g,
                                            train_nid.cuda(),
                                            sampler,
                                            device="cuda",
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            use_ddp=True,
                                            use_uva=True)

    tic = time.time()
    presampling_heat = torch.zeros((g.num_nodes(), ), dtype=torch.float32)
    sampling_times = 0
    for epoch in range(args.num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            torch.cuda.synchronize()
            presampling_heat[input_nodes.cpu()] += 1
            sampling_times += 1
    presampling_heat = presampling_heat / sampling_times
    toc = time.time()

    presampling_heat_accessed = presampling_heat[presampling_heat > 0]
    print(
        "Presampling done, max: {:.3f} min: {:.3f} avg: {:.3f}, time: {:.3f} s"
        .format(
            torch.max(presampling_heat_accessed).item(),
            torch.min(presampling_heat_accessed).item(),
            torch.mean(presampling_heat_accessed).item(), toc - tic))

    save_fn = os.path.join(
        args.save_path,
        args.dataset + "_" + args.fan_out + "_presampling_heat.pt")
    torch.save(presampling_heat, save_fn)
    print("Result saved to {}".format(save_fn))
