from data_utils import *
from model import *
from utils import *
import numpy as np

if __name__ == '__main__':
    # Do not modify this cell.
    feature_dim = 128
    temperature = 0.5 # tau
    k = 200
    batch_size = 8
    num_workers = 4
    epochs = 150
    percentage = 0.5
    pretrained_path = 'pretrained_model/pretrained_simclr_cifar10.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data.
    train_transform = compute_train_transform()
    train_data = BarrierReefPair(root='great-barrier-reef-small', train=True, transform=train_transform)
    # train_data = torch.utils.data.Subset(train_data, list(np.arange(int(len(train_data) * percentage))))
    print(batch_size)
    train_loader = DataLoader(train_data, batch_size=batch_size // 2, shuffle=True, num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_transform = compute_test_transform()
    # memory_data = BarrierReefPair(root='tensorflow-great-barrier-reef', train=True, transform=test_transform, download=True)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = BarrierReefPair(root='great-barrier-reef-small', train=False, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Set up the model and optimizer config.
    model = Model(feature_dim)
    # model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),)) # a tool to count flops
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # c = len(memory_data.classes)
    print()

    # Training loop.
    results = {'train_loss': [], 'test_loss': []}  # << -- output

    # create path "results"
    if not os.path.exists('results'):
        os.mkdir('results')

    # run training
    # best_acc = 0.0
    best_loss = 10.0**10
    from tqdm import tqdm
    for epoch in range(1, epochs + 1):
        train_loss = train(
            model, train_loader, optimizer, epoch, epochs, batch_size=batch_size, temperature=temperature, device=device
        )
        results['train_loss'].append(train_loss)
        test_loss = train(
            model, test_loader, None, epoch, epochs, batch_size=batch_size, temperature=temperature, device=device
        )
        results['test_loss'].append(test_loss)
        # Save statistics.
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'pretrained_model/trained_simclr_model.pth')
    print(results)
