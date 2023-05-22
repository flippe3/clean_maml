    # Create a vanilla PyTorch neural network.
    # inplace_relu = True
    # net = nn.Sequential(
    #     nn.Conv2d(1, 64, 3),
    #     nn.BatchNorm2d(64, affine=True, track_running_stats=False),
    #     nn.ReLU(inplace=inplace_relu),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, affine=True, track_running_stats=False),
    #     nn.ReLU(inplace=inplace_relu),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, affine=True, track_running_stats=False),
    #     nn.ReLU(inplace=inplace_relu),
    #     nn.MaxPool2d(2, 2),
    #     nn.Flatten(),
    #     nn.Linear(64, args.n_way)).to(device)