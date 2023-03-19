from skimage.measure import label
import numpy as np
import torch
from models import U_net_spec,U_net_RGB,DiceBCELoss
from tqdm import tqdm
import sys

def AJI(target,pred):
    pred_label = label(pred)
    target_label = label(target)
    C = 0
    U = 0
    Unused = np.ones(pred_label.max()-1)
    for i in range(1,max(np.unique(target_label))+1):
        options = np.unique(pred_label[(i==target_label) & (pred_label!=0)])
        cur_score = 0
        if np.shape(options)[0]>0:
            for j in options:
                new_score = np.sum((i==target_label) & (j==pred_label)) / np.sum((i==target_label) | (j==pred_label))
                if new_score>cur_score:
                    cur_score = new_score
                    cur_index = j
            C = C + np.sum((i==target_label) & (j==pred_label))
            U = U + np.sum((i==target_label) | (j==pred_label))
            Unused[cur_index-1] = 0
    for i in range(len(Unused)):
        if Unused[i] == 1:
            U = U + np.sum(pred_label==(i+1))


    return C/U



def full_forward(sample_size,model,image,device):
    with torch.no_grad():
        X = list(range(0,image.shape[0]-sample_size,int(sample_size/2)))
        X.append(image.shape[0]-sample_size)
        Y = list(range(0,image.shape[1]-sample_size,int(sample_size/2)))
        Y.append(image.shape[1]-sample_size)
        mask = torch.zeros((image.shape[0],image.shape[1]))
        model.eval()
        for x,y  in [(x,y) for x in X for y in Y]:

            input = image[x:x+sample_size,y:y+sample_size,:]
            input = torch.permute(input,(2,0,1))
            input = torch.reshape(input,(1,input.size(0),input.size(1),input.size(2)))
            input = input.to(device=device, dtype=torch.float)
            output = model(input)
            output = torch.squeeze(output.detach().cpu())


            n = int(sample_size/4)*(x!=0)*(x!=X[-1]) + (sample_size-(image.shape[0]-X[-2]-int(3*sample_size/4)))*(x==X[-1])
            m = int(3*sample_size/4)*(x!=X[-1]) + sample_size*(x==X[-1])
            u = int(sample_size/4)*(y!=0)*(y!=Y[-1]) + (sample_size-(image.shape[1]-Y[-2]-int(3*sample_size/4)))*(y==Y[-1])
            v = int(3*sample_size/4)*(y!=Y[-1]) + sample_size*(y==Y[-1])

            mask[x+n:x+m,y+u:y+v] = output[
                (x!=X[-1])*(x!=0)*int(sample_size/4)+(x==X[-1])*(sample_size-m+n):sample_size-int(sample_size/4)*(x!=X[-1]),
                (y!=Y[-1])*(y!=0)*int(sample_size/4)+(y==Y[-1])*(sample_size-v+u):sample_size-int(sample_size/4)*(y!=Y[-1])]>0.5
    return mask


