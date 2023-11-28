import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heat_fn", type=str)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    print(args)
    heat = torch.load(args.heat_fn)
    num_steps = int(1 / args.step + 1)
    init = 0.0
    hot_num = (heat > 0).sum()
    for i in range(num_steps):
        num = (heat > init).sum()
        rate = num / heat.shape[0]
        true_rate = num / hot_num
        print("{:6.3f} {:10d} {:6.3f} {:6.3f}".format(init, num, rate,
                                                      true_rate))
        init += args.step
