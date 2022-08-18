import os
from tqdm import tqdm
from blurtest import *

image_root = './data/FFHQ_1024/clean/'

look_upto = 0
for file in os.listdir(image_root):
    if os.path.splitext(file)[-1] not in ['.png', '.jpg']:
        continue

    look_upto += 1
    image_name = os.path.splitext(file)[0]
    os.makedirs(os.path.join(image_root, image_name), exist_ok=True)
    clean_img = cv2.imread(os.path.join(image_root, file))
    print(f"Creating blur images [{look_upto}/30]")
    for i in tqdm(range(101)):
        blurred_img, _ = blurring(clean_img, d=i, angle=45)
        cv2.imwrite(os.path.join(image_root, image_name, f'fix_{i}.png'), blurred_img)

    if look_upto == 30:
        break
