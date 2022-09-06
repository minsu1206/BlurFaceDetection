import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LabelDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root        # clean image directory after create_blur_only_img
        self.img_names = os.walk(self.root)

    def __len__(self):
        return len(self.img_names)

    def _np2torch(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def __call__(self, idx):
        clean_img_path = self.root + self.img_names[idx]
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
        print(len(images))  # debugging
        return images, paths


if __name__ == "__main__":
    from facenet_pytorch import MTCNN, InceptionResnetV1
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    label_dataset = LabelDataset(args.root)

    label_dataloader = DataLoader(label_dataset, batch_size=args.batch, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # FIXME
    # dataloader 올리는게 맞나? 필요없을 것 같은데
    for i, (img, path) in tqdm(enumerate(label_dataloader)):
        print(img.shape)
        print(path.shape)
        break



    
        
