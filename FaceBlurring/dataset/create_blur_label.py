import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import torchvision
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

class LabelDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root        # clean image directory after create_blur_only_img
        # print(type(self.root))
        self.img_names = os.listdir(self.root)

    def __len__(self):
        return len(self.img_names)

    def _np2torch(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def __getitem__(self, idx):
        # print(self.img_names[idx])
        # print(self.root)

        clean_img_path = self.root + '/' + self.img_names[idx]
        # print(self.img_names[idx])
        # print(cv2.imread(clean_img_path, cv2.IMREAD_COLOR).shape)

        clean_img = cv2.cvtColor(
            cv2.imread(clean_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        clean_img = self._np2torch(clean_img)
        images = [clean_img]
        paths = [clean_img_path]
        blur_count = 0
        blur_img_path = clean_img_path.replace('clean', 'blur')
        blur_img_path = blur_img_path.replace('.png', f'_{blur_count}.png')
        while os.path.exists(blur_img_path):
            blur_img = cv2.cvtColor(
                cv2.imread(blur_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            blur_img = self._np2torch(blur_img)
            images.append(blur_img)
            paths.append(blur_img_path)
            new_count = blur_count + 1
            blur_img_path = blur_img_path.replace(f'_{blur_count}.png', f'_{new_count}.png')
            blur_count += 1

        images = torch.cat(images, 0)
        # print(len(images))  # debugging
        return images, paths, len(images)


if __name__ == "__main__":
    from facenet_pytorch import MTCNN, InceptionResnetV1
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/FFHQ_512/clean/ffhq')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.root.replace('clean', 'label'), exist_ok=True)
    label_dataset = LabelDataset(args.root)

    print(len(label_dataset))

    # label_dataloader = DataLoader(label_dataset, batch_size=args.batch, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    assert len(label_dataset) % args.batch == 0, "WRONG Batch size"
    # batch 2나 4 권장
    for idx in range(int(len(label_dataset) // args.batch)):

        batch_imgs = []
        batch_paths = []
        batch_marks = []
        for i in range(args.batch):
            imgs, paths, marks = label_dataset[idx + i]
            batch_imgs.append(imgs)
            batch_paths.append(paths)
            batch_marks.append(marks)

        batch_imgs = torch.cat(batch_imgs).to(device)
        batch_output = model(batch_imgs)

        last_mark = 0
        for paths, marks in zip(batch_paths, batch_marks):
            unit = batch_output[last_mark:marks+last_mark]
            last_mark = marks

            clean = unit[0]
            blurs = unit[1:]

            cossim_list = []
            # unit of Batch - Cosine Similarity
            cossims = F.cosine_similarity(clean, blurs)
            cossims = cossims.cpu().detach()
            for cossim in cossims:
                cossim_list.append(round(cossim.item(), 5))
            
            for cos, path in zip(cossim_list, paths[1:]):
                txt_path = path.replace('blur', 'label').replace('.png', '.txt')
                with open(txt_path, 'w') as f:
                    f.write(str(cos))
        break

                    

    

    
        
