
def load_data(directory_name,img_size = 128):
    from scipy import io
    import pickle
    import os
    import torch
    import numpy as np
    import re


    # Convert the mat files to picle files
    for i, file_name in enumerate(os.listdir(directory_name)):
        if file_name.endswith('.mat') & ~os.path.exists(directory_name + file_name.replace('mat', 'pickle')):
            mat = io.loadmat(directory_name + file_name)
            spec_full = np.array(mat['spec'])
            RGB_full = np.array(mat['RGB'])
            RGB_balance_full = np.array(mat['RGB_balance'])
            mask_full = np.array(mat['mask'])
            image_size = mask_full.shape


            X = list(range(0, image_size[0] - 128, 64))
            X.append(image_size[0] - 128)
            Y = list(range(0, image_size[1] - 128, 64))
            Y.append(image_size[1] - 128)

            spec = np.array([spec_full[x:x + 128, y:y + 128] for x in X for y in Y])
            RGB = np.array([RGB_full[x:x + 128, y:y + 128] for x in X for y in Y])
            RGB_balance = np.array([RGB_balance_full[x:x + 128, y:y + 128] for x in X for y in Y])
            mask = np.array([mask_full[x:x + 128, y:y + 128] for x in X for y in Y])

            filename = directory_name + file_name.replace('mat', 'pickle')
            with open(filename, 'wb') as outfile:
                pickle.dump({'spec': spec, 'RGB': RGB, 'mask': mask, 'RGB_balance': RGB_balance},
                            outfile)
                outfile.close()
    # load the picle data
    first = True
    for i, file_name in enumerate(os.listdir(directory_name)):
        if file_name.endswith('.pickle'):
            print('Load:' , file_name)
            with open(directory_name + file_name, 'rb') as f:
                pikle_file = pickle.load(f)
                cur_spec = torch.tensor(pikle_file['spec'])
                cur_RGB = torch.tensor(pikle_file['RGB'])
                cur_mask = torch.tensor(pikle_file['mask'])
                cur_RGB_balance = torch.tensor(pikle_file['RGB_balance'])
                cur_ID = int(re.findall(r'\d+', file_name)[0])
                f.close()
            if (first):
                spec = cur_spec
                RGB = cur_RGB
                mask = cur_mask
                RGB_balance = cur_RGB_balance
                torch.ones(cur_RGB_balance.size()[0])
                ID = cur_ID * torch.ones(cur_RGB_balance.size()[0])
                first = False
            else:
                spec = torch.cat((spec, cur_spec))
                RGB = torch.cat((RGB, cur_RGB))
                mask = torch.cat((mask, cur_mask))
                RGB_balance = torch.cat((RGB_balance, cur_RGB_balance))
                ID = torch.cat((ID, cur_ID * torch.ones(cur_RGB_balance.size()[0])))

    return spec,RGB,mask,RGB_balance,ID


