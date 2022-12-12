from data_utils import *
from model import *
from utils import *
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    # Do not modify this cell.
    feature_dim = 128
    temperature = 0.5
    k = 200
    batch_size = 8
    epochs = 50
    percentage = 0.1
    pretrained_path = './pretrained_model/trained_simclr_model_8.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_transform = compute_detect_transform
    train_data = BarrierReefDetect(root='great-barrier-reef-small', train=True, transform=det_transform)
    # train_data = torch.utils.data.Subset(train_data, list(np.arange(int(len(train_data) * percentage))))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    # test_transform = compute_test_transform()
    test_data = BarrierReefDetect(root='great-barrier-reef-small', train=False, transform=det_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Detector()
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
    model = model.to(device)
    for param in model.f.parameters():
        param.requires_grad = False

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # optimizer = optim.Adam(model.fc.parameters(), lr=1e-5, weight_decay=1e-6)
    print()

    # Training loop.
    pretrain_results = {'train_loss': [], 'train_mAP_0.1': [], 'train_mAP_0.5': [],
                        'test_loss': [], 'test_mAP_0.1': [], 'test_mAP_0.5': []}
    # create path "results"
    if not os.path.exists('results'):
        os.mkdir('results')

    best_loss = 10**4
    for epoch in range(1, epochs + 1):
        lr = 1e-4 if epoch <= 2 else 1e-6 if epoch <= 10 else 1e-8
        optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-6)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        train_loss, train_mAP_01, train_mAP_05 = train_val(model, train_loader, optimizer, epoch, epochs, device)
        pretrain_results['train_loss'].append(train_loss)
        pretrain_results['train_mAP_0.1'].append(train_mAP_01)
        pretrain_results['train_mAP_0.5'].append(train_mAP_05)
        test_loss, test_mAP_01, test_mAP_05 = train_val(model, test_loader, None, epoch, epochs)
        pretrain_results['test_loss'].append(test_loss)
        pretrain_results['test_mAP_0.1'].append(test_mAP_01)
        pretrain_results['test_mAP_0.5'].append(test_mAP_05)
        print(pretrain_results)
        if test_loss < best_loss:
            best_loss = test_loss
            best_mAP_01 = test_mAP_01
            best_mAP_05 = test_mAP_05

    # Print the best test accuracy. You should see a best top-1 accuracy of >=70%.
    print('Best mAP@0.1 with self-supervised learning: ', best_mAP_01)
    print('Best mAP@0.5 with self-supervised learning: ', best_mAP_05)

    print(pretrain_results)
