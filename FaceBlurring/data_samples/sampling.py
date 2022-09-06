import os
import pandas as pd
import cv2
from tqdm import tqdm
import pickle
from repeat_blur import *

if __name__ == '__main__':
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    path_n_label = pd.DataFrame.to_dict(pd.read_csv('/data/faceblur/BlurFaceDetection/FaceBlurring/data/label_blur_defocus/label/data_label.csv'))
    sampling_src = []
    cnt=0
    for i in range(len(path_n_label['filename'])):
        if not path_n_label['train'][i]:
            clean_path = (os.path.sep).join(path_n_label['filename'][i].split(os.path.sep)[:3])+'/clean/'+\
                         (os.path.sep).join(path_n_label['filename'][i].split(os.path.sep)[4:])
            sampling_src += [clean_path]
            cnt += 1
            if cnt==30:
                break
    dict_for_label = {'filename': [], 'cosine': []}
    os.makedirs('./samples/', exist_ok=True)
    for pth in tqdm(sampling_src):
        sample_path = os.path.join('./samples', os.path.splitext(pth.split(os.path.sep)[5])[0])
        os.makedirs(sample_path, exist_ok=True)
        sample = cv2.imread(pth)
        for i in range(101):
            blurred1 = blurring(sample, i)
            cv2.imwrite(os.path.join(sample_path, f'random_{i}.png'), blurred1)


            blurred2 = blurring(sample, i, 45)
            cv2.imwrite(os.path.join(sample_path, f'fix_{i}.png'), blurred2)

            emb_clean = resnet(torch.Tensor(sample).permute(2, 0, 1).unsqueeze(0).to(device))
            emb_blur1 = resnet(torch.Tensor(blurred1).permute(2, 0, 1).unsqueeze(0).to(device))
            emb_blur2 = resnet(torch.Tensor(blurred2).permute(2, 0, 1).unsqueeze(0).to(device))
            degree1 = 1 - F.cosine_similarity(emb_clean, emb_blur1, 1).item()
            degree2 = 1 - F.cosine_similarity(emb_clean, emb_blur2, 1).item()

            dict_for_label['filename'].append(os.path.abspath(os.path.join(sample_path, f'random_{i}.png')))
            dict_for_label['cosine'].append(degree1)

            dict_for_label['filename'].append(os.path.abspath(os.path.join(sample_path, f'fix_{i}.png')))
            dict_for_label['cosine'].append(degree2)

    df = pd.DataFrame(dict_for_label)
    df.to_csv(os.path.join("../data/label_val.csv"))
    '''
    info = pd.DataFrame.to_dict(pd.read_csv('/data/faceblur/BlurFaceDetection/FaceBlurring/data/label_val.csv'))
    random_mean = np.zeros(100)
    fix_mean = np.zeros(100)
    for i in range(len(info['filename'])):
        if i%202 == 0 or i%202 == 1:
            continue

        if i%2:
            fix_mean[(i%202-1)//2-1] += info['cosine'][i]
        else:
            random_mean[(i%202)//2-1] += info['cosine'][i]

    random_mean/=30
    fix_mean /= 30

    with open('random_reference.pkl', 'wb') as f:
        pickle.dump(random_mean, f, pickle.HIGHEST_PROTOCOL)

    with open('fix_reference.pkl', 'wb') as f:
        pickle.dump(fix_mean, f, pickle.HIGHEST_PROTOCOL)
