import os
from tqdm import tqdm
from blur import *

image_root = './data/FFHQ_1024/clean/00595.png'

#look_upto = 0
'''
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
'''

image = cv2.imread(image_root)
blur1, _ = blurring(image, 78, 90)
blur2, _ = blurring(image, 97, 30)

cv2.imwrite('./data/FFHQ_1024/clean/00595/random_78.png', blur1)
cv2.imwrite('./data/FFHQ_1024/clean/00595/random_97.png', blur2)
