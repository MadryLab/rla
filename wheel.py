import torch


if __name__ == '__main__':
    while True:
        rand = torch.rand(100000)
        min_i = rand[0]
        for i in rand:
            min_i = min(i, min_i)