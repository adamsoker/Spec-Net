

def fit_model(train_dataloader,valid_dataloader,model,criterion,device,epochs=100,lr = 0.01,step_size=20,gamma=0.90,DisEpochSum=True,break_cond=True):
    from tqdm import tqdm
    import sys
    import numpy as np
    import torch
    from torchmetrics.classification import BinaryF1Score
    metric = BinaryF1Score().to(device=device)

    net =model
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_loss_history = []
    train_accuracy_history = []
    valid_loss_history = []
    valid_accuracy_history = []
    valid_f1_history = []
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        epoch_loss = 0
        accuracy   = 0
        net.train()
        pixel_num = 0
        with tqdm(total=len(train_dataloader), file=sys.stdout) as pbar:
            for i, (input_spec, input_label) in enumerate(train_dataloader):
                pbar.set_description('Epoch: %d/%d' % (epoch+1,epochs))
                pbar.update(1)
                input_spec = input_spec.to(device=device, dtype=torch.float)
                input_spec = torch.permute(input_spec,(0,3,1,2))
                input_label = input_label.to(device=device, dtype=torch.float)
                output = net(input_spec)
                loss = criterion(output.squeeze(), input_label.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss + loss.item()
                pred = (output.view(-1)>0.5)
                accuracy = accuracy + torch.sum(pred == input_label.view(-1))
                pixel_num = pixel_num + len(pred)

        train_loss_history.append(epoch_loss/pixel_num)
        train_accuracy_history.append(accuracy/pixel_num)
        scheduler.step()

        net.eval()
        with torch.no_grad():
            accuracy = 0
            epoch_loss = 0
            f1 = 0
            pixel_num = 0
            for i, (input_spec, input_label) in enumerate(valid_dataloader, 0):
                input_spec = input_spec.to(device=device, dtype=torch.float)
                input_spec = torch.permute(input_spec,(0,3,1,2))
                output= net(input_spec)
                input_label = input_label.to(device=device, dtype=torch.float)
                loss = criterion(output.squeeze(), input_label.squeeze())
                epoch_loss = loss.item() + epoch_loss
                pred = (output.view(-1)>0.5)
                accuracy = accuracy + torch.sum(pred == input_label.view(-1))
                pixel_num = pixel_num + len(pred)
                f1 = f1 + metric(pred,input_label.view(-1))
            valid_accuracy_history.append(accuracy/pixel_num)
            valid_loss_history.append(epoch_loss/pixel_num)
            valid_f1_history.append(f1/(i+1))


    spec_history = (torch.Tensor(train_loss_history),torch.Tensor(valid_loss_history),torch.Tensor(train_accuracy_history),torch.Tensor(valid_accuracy_history),torch.Tensor(valid_f1_history))
    return net, spec_history
