import torch
from tqdm import tqdm


def train(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0

    for bags, labels, ab_bags, ab_labels in tqdm(data_loader):
        data = torch.cat((bags, ab_bags), dim=1).to(device)
        batch_size = data.shape[0]
        data = data.view(-1, data.size(-1)).to(device)

        outputs = model(data)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    return train_loss / len(data_loader) ,optimizer.param_groups[0]['lr']


def test(model, test_loader):
    pass
