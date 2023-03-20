Imports


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from load_data import load_data
from dataset import Full_Dataset,RGB_Dataset ,spec_Dataset
from models import U_net_spec,U_net_RGB,DiceBCELoss,binary_Loss
from fit_model import fit_model
from tqdm import tqdm
import sys
from scipy import io
import os
from function import full_forward, AJI
```

Device definition


```python
device = torch.device('cuda:0')
```

Creat pickle files from the mat files and load them.
You may request the database by email adam-soker@campus.technion.ac.il .


```python
Train_data = './Train_data/'
spec,RGB,mask,RGB_balance,ID = load_data(directory_name = Train_data)
```

Cross validation


```python
torch.cuda.empty_cache()
p=0 ## The dropout factor. was not used in the article.
N = 2 ## A factor for determining the size of the model. was set to 2 in the article.
Lambda = [0.5,1,2] ## Lambda values (as listed in the article) that are checked
epochs = 6

## Learning rate settings
lr = 1e-4
step_size=20
batch_size = 16
gamma=0.90



ID_unique = np.unique(ID)
summery =[]
first = True
for i in range(len(ID_unique)):
    for chosen_model in ['spec','RGB']:
        if 'RGB' in chosen_model:
            RGB_mean = RGB[ID!=ID_unique[i]].mean(0).mean(0).mean(0)
            RGB_norm = (RGB-RGB_mean)
            RGB_std = (RGB_norm[ID!=ID_unique[i]]*RGB_norm[ID!=ID_unique[i]]).mean(0).mean(0).mean(0)
            RGB_norm = RGB_norm/RGB_std

            net = U_net_RGB
            train_dataset = RGB_Dataset(RGB_norm[ID!=ID_unique[i]].to(device=device, dtype=torch.float),mask[ID!=ID_unique[i]].to(device=device, dtype=torch.float))
            valid_dataset = RGB_Dataset(RGB_norm[ID==ID_unique[i]].to(device=device, dtype=torch.float),mask[ID==ID_unique[i]].to(device=device, dtype=torch.float))
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
        else:
            spec_mean = spec[ID!=ID_unique[i]].mean(0).mean(0).mean(0)
            spec_norm = (spec-spec_mean)
            spec_std = (spec_norm[ID!=ID_unique[i]]*spec_norm[ID!=ID_unique[i]]).mean(0).mean(0).mean(0)
            spec_norm = spec_norm/spec_std

            net = U_net_spec
            train_dataset = spec_Dataset(spec_norm[ID!=ID_unique[i]].to(device=device, dtype=torch.float),mask[ID!=ID_unique[i]].to(device=device, dtype=torch.float))
            valid_dataset = spec_Dataset(spec_norm[ID==ID_unique[i]].to(device=device, dtype=torch.float),mask[ID==ID_unique[i]].to(device=device, dtype=torch.float))
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=False)


        for j,L in enumerate(Lambda):
            criterion=DiceBCELoss(Lambda=L)
            model = net(p=p,N=N).cuda()
            print('Start {} Lambda: {}  iteration: {}'.format(chosen_model.replace('_',' '),L,i+1))
            history = list([0,0,0,0,0])

            # exec('model = {}(p=p,N=N).cuda()'.format(chosen_model))
            fit, cur_history = fit_model(train_dataloader,valid_dataloader,model=model,criterion=criterion,device=device,epochs=epochs,lr = lr,step_size=step_size,gamma=gamma,DisEpochSum=False,break_cond=False)
            cur_history = torch.stack(cur_history)
            exec("if first:\n   history_{}_{} = cur_history\nelse:\n  history_{}_{}=history_{}_{}+cur_history".format(chosen_model,j,chosen_model,j,chosen_model,j,chosen_model,j))
            print('End {} Lambda: {}  iteration: {}\n'.format(chosen_model.replace('_',' '),L,i+1))
        first = False



```

Plot the validation result


```python
for chosen_model in ['spec','RGB']:
     for j,L in enumerate(Lambda):
         exec("history = history_{}_{}".format(chosen_model,j))
         summery.append({'lr':lr,'chosen_model':chosen_model,'tr loss':history[0]/len(ID_unique),'val loss':history[1]/len(ID_unique),'tr acc':history[2]/len(ID_unique),'val acc':history[3]/len(ID_unique),'val F1':history[4]/len(ID_unique),'Lambda':L})

chosen_model = []
```


```python
fig, ax = plt.subplots(int(len(summery)/2),3, figsize=(40, 40))
for i in range(int(len(summery)/2)):
    for j,mode in enumerate(['SI','RGB']):
        ax[i,0].plot(summery[2*i+j]['val loss'],label= 'Valid Loss {}'.format(mode))
        ax[i,1].plot(summery[2*i+j]['val acc'],label= 'Valid Accuracy {}'.format(mode))
        ax[i,2].plot(summery[2*i+j]['val F1'],label= 'Valid Accuracy {}'.format(mode))

    ax[i,0].grid()
    ax[i,0].legend(fontsize=20)
    ax[i,0].set_ylabel('loss',fontsize=20)
    ax[i,0].set_xlabel('Epoch',fontsize=20)
    ax[i,0].set_title('Loss, Lambda = {}'.format(summery[2*i+j]['Lambda']), fontsize=30, color='k')

    ax[i,1].grid()
    ax[i,1].legend(fontsize=20)
    ax[i,1].set_ylabel('Accuracy',fontsize=20)
    ax[i,1].set_xlabel('Epoch',fontsize=20)
    ax[i,1].set_title('Accuracy, Lambda = {}'.format(summery[2*i+j]['Lambda']), fontsize=30, color='k')

    ax[i,2].grid()
    ax[i,2].legend(fontsize=20)
    ax[i,2].set_ylabel('F1-score',fontsize=20)
    ax[i,2].set_xlabel('Epoch',fontsize=20)
    ax[i,2].set_title('F1-score, Lambda = {}'.format(summery[2*i+j]['Lambda']), fontsize=30, color='k')
```

Model sizes


```python
import models
importlib.reload(models)
from models import U_net_spec,U_net_RGB,DiceBCELoss

model = U_net_RGB(N=2,p=0)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('U-net RGB',pytorch_total_params)


model = U_net_spec(N=2,p=0)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('U-net SI',pytorch_total_params)
```

Training the final spec-net model.



```python
spec_mean = spec.mean(0).mean(0).mean(0)
spec_norm = (spec-spec_mean)
spec_std = (spec_norm*spec_norm).mean(0).mean(0).mean(0)
spec_norm = spec_norm/spec_std


train_dataset = spec_Dataset(spec_norm.to(device=device, dtype=torch.float),mask.to(device=device, dtype=torch.float))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

epochs = 100
p = 0
N = 2
L = 0.5

net_spec = U_net_spec(p=p,N=N).cuda()
optimizer = torch.optim.Adam(net_spec.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
net_spec.train()


criterion=DiceBCELoss(Lambda=L)
for epoch in range(epochs):
    with tqdm(total=len(train_dataloader), file=sys.stdout) as pbar:
        for i, (input, input_label) in enumerate(train_dataloader,0):
            pbar.set_description('Epoch %d/%d' % (1 + epoch, epochs))

            pbar.update(1)
            input = input
            input = torch.permute(input,(0,3,1,2))
            input_label = input_label
            output = net_spec(input)
            output = output
            loss = criterion(output.squeeze(), input_label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

net_spec.eval()
PATH = './models/spec_p{}_L{}_epoch{}.pt'.format(N, p, L, int(epoch + 1))
torch.save(net_spec.state_dict(), PATH)

```

Training the final RGB-net model.


```python
RGB_mean = RGB.mean(0).mean(0).mean(0)
RGB_norm = (RGB-RGB_mean)
RGB_std = (RGB_norm*RGB_norm).mean(0).mean(0).mean(0)
RGB_norm = RGB_norm/RGB_std


train_dataset = RGB_Dataset(RGB_norm.to(device=device, dtype=torch.float),mask.to(device=device, dtype=torch.float))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

epochs = 100
p = 0
N = 2
L = 2

net_RGB = U_net_RGB(p=p,N=N).cuda()
optimizer = torch.optim.Adam(net_RGB.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
net_RGB.train()


criterion=DiceBCELoss(Lambda=L)
for epoch in range(epochs):
    with tqdm(total=len(train_dataloader), file=sys.stdout) as pbar:
        for i, (input, input_label) in enumerate(train_dataloader,0):
            pbar.set_description('Epoch %d/%d' % (1 + epoch, epochs))
            pbar.update(1)
            input = input
            input = torch.permute(input,(0,3,1,2))
            input_label = input_label
            output = net_RGB(input)
            output = output
            loss = criterion(output.squeeze(), input_label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

net_RGB.eval()
PATH = '../models/RGB_p{}_L{}_epoch{}.pt'.format(N, p, Lambda, int(epoch + 1))
torch.save(net_RGB.state_dict(), PATH)
```

Testing


```python

Test_dir = './Test_data'

mask_spec_pred = []
F1_spec = []
AJI_spec = []


mask_RGB_pred = []
F1_RGB = []
AJI_RGB = []

from torchmetrics.classification import BinaryF1Score
F1_metric = BinaryF1Score().to(device=device)

for i, file_name in enumerate(os.listdir(Test_dir)):
    if file_name.endswith('.mat') :
        mat = io.loadmat(Test_dir +'/'+ file_name)
        mask_gt = mat['mask']

        spec_test = ((torch.tensor(mat['spec'])-spec_mean)/spec_std).to(device=device, dtype=torch.float)
        mask_spec_pred_cur = full_forward(sample_size=128,model=net_spec,image=spec_test,device=device)
        mask_spec_pred.append(mask_spec_pred_cur)
        F1_spec.append(F1_metric(mask_spec_pred_cur,mask_spec_pred_cur))
        AJI_spec.append(AJI(mask_gt,mask_spec_pred_cur))
        
        RGB_test = ((torch.tensor(mat['RGB'])-RGB_mean)/RGB_std).to(device=device, dtype=torch.float)
        mask_RGB_pred_cur = full_forward(sample_size=128,model=net_RGB,image=RGB_test,device=device)
        mask_RGB_pred.append(mask_RGB_pred_cur)
        F1_RGB.append(F1_metric(mask_RGB_pred_cur,mask_RGB_pred_cur))
        AJI_RGB.append(AJI(mask_gt,mask_RGB_pred_cur))
```
