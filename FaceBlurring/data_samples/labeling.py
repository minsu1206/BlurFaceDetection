import pandas as pd
import os, cv2
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    dict_for_label = {'filename' : [], 'cosine' : []}
    root = '/data/faceblur/BlurFaceDetection/FaceBlurring/data_samples/data/FFHQ_1024/clean'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    for r in os.listdir(root):
        src = os.path.join(root, r, 'fix_0.png')
        for i in tqdm(range(101)):
            blr_fix = os.path.join(root, r, f'fix_{i}.png')
            blr_rand = os.path.join(root, r, f'random_{i}.png')

            clean, blr1, blr2 = cv2.imread(src), cv2.imread(blr_fix), cv2.imread(blr_rand)
            emb_clean = resnet(torch.Tensor(clean).permute(2, 0, 1).unsqueeze(0).to(device))
            emb_blur1 = resnet(torch.Tensor(blr1).permute(2, 0, 1).unsqueeze(0).to(device))
            emb_blur2 = resnet(torch.Tensor(blr2).permute(2, 0, 1).unsqueeze(0).to(device))
            degree1 = 1-F.cosine_similarity(emb_clean, emb_blur1, 1).item()
            degree2 = 1-F.cosine_similarity(emb_clean, emb_blur2, 1).item()

            dict_for_label['filename'] += [blr_fix]
            dict_for_label['filename'] += [blr_rand]
            dict_for_label['cosine'] += [degree1]
            dict_for_label['cosine'] += [degree2]


    df = pd.DataFrame(dict_for_label)
    df.to_csv(os.path.join("../data/label_val.csv"))

