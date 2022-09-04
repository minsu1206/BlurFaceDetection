import copy

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from blur import *

def iterative_blur(model, image, blur_method=''):
    # model = face recognition model
    # blur method
        # deblurGAN
        # defocus

    clean_image = image
    clean_image_tensor = torch.Tensor(clean_image).permute(2, 0, 1).unsqueeze(0).cuda()

    is_adversarial = False

    if blur_method == 'defocus':
        blur_image, blur_image_tensor = defocus_blur_func(clean_image)
        clean_result = model(clean_image_tensor)

        while not is_adversarial:

            # clean_embed = model(clean_image)
            # blur_emded = model(blur_image)

            #blur_result = model(blur_image)

            #if clean_result != blur_result:
                #is_adversarial = True
                #return blur_image

            cossim = cosine_similarity_func(clean_image, blur_image)
            if cossim < 0.4:
              is_adversarial = True
              return blur_image
            
            blur_image, blur_image_tensor = defocus_blur_func(blur_image)


    if blur_method == 'deblurGAN':
        blur_image, blur_image_tensor = deblurGAN_blur_func(clean_image)
        clean_result = model(clean_image_tensor)
        
        while not is_adversarial:

            # clean_embed = model(clean_image)
            # blur_emded = model(blur_image)

            #blur_result = model(blur_image)

            #if clean_result != blur_result:
                #is_adversarial = True
                #return blur_image

            cossim = cosine_similarity_func(clean_image, blur_image)
            if cossim < 0.4:
              is_adversarial = True
              return blur_image
            
            blur_image, blur_image_tensor = deblurGAN_blur_func(blur_image)

def iterative_blur_n(model, image, n=3, dsize=(112, 112), blur_method_list=['deblurGAN', 'defocus'], device='cpu'):
    '''
    get n blur images list
    Args:
        image: clean image you want to get blurred images
        n: iterative number
        dsize: resize size of output image
        blur_method_list
    Returns:
        clean_image: original Image
        blur_image_list
        cossim_list
    '''
    # [9/3 수정] -> process 최적화
    # (1) redundant model build -> remove!
    # (2) embedding -> batch forwarding!

    clean_image = image
    clean_img_tensor = torch.from_numpy(clean_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    # clean_img_embed = model(clean_img_tensor).squeeze(0).cpu().detach()
    blur_image = copy.deepcopy(image)
    image_tensor_list, gen_image_list, cossim_list = [clean_img_tensor], [], []

    for count in range(1, n+1):
        blur_method = np.random.choice(blur_method_list)
        if blur_method == 'defocus':
            blur_image, blur_image_tensor = defocus_blur_func(blur_image)

        elif blur_method == 'deblurGAN':
            blur_image, blur_image_tensor = deblurGAN_blur_func(blur_image)

        gen_blur_image = cv2.resize(blur_image, dsize=dsize, interpolation=cv2.INTER_AREA)
        image_tensor_list.append(blur_image_tensor)
        gen_image_list.append(gen_blur_image)

    batch_imgs = torch.cat(image_tensor_list, dim=0)
    # print(batch_imgs.shape) ok

    cossim_list = cosine_similarity_batch(model, batch_imgs)
    print(cossim_list)
    clean_image = cv2.resize(clean_image, dsize=dsize, interpolation=cv2.INTER_AREA)
    return clean_image, gen_image_list, cossim_list
    
def defocus_blur_func(img):
    #input image (doesn't matter whether clean or blur)
    #return blurred image when the input is in.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blur_img, blur_deg = blurring(img, {'mean':25, 'var':10, 'dmin':0, 'dmax':50})
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).to(device)
    return blur_img, blur_img_tensor


def deblurGAN_blur_func(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trajectory = Trajectory({'canvas':64, 'iters':2000, 'max_len':60, 'expl':0.001}).fit()
    psf, degree = PSF(64, trajectory).fit()
    part_parameters = np.random.choice([1, 2, 3])
    clean_img, blur_img = BlurringImage(img, psf, part_parameters).blur_image()
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).to(device)
    return blur_img, blur_img_tensor


def cos_sim(emb1, emb2):
	return np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))

def cosine_similarity_func(model, clean_embed, blur_img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blur_img_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    blur_img_embed = model(blur_img_tensor).squeeze(0).cpu().detach()
    cossim = cos_sim(clean_embed, blur_img_embed)
    return cossim

def cosine_similarity_batch(model, imgs):
    img_embed = model(imgs)
    clear_img_embed = img_embed[0, :]     # (F)
    blur_imgs_embed = img_embed[1:, :]    # (B-1, F)

    cossim_list = []
    # Batch Cosine Similarity (Not on CPU. CUDA is faster. Detach follows CUDA matrix operation)

    cossims = torch.nn.functional.cosine_similarity(clear_img_embed, blur_imgs_embed)
    cossims = cossims.cpu().detach()
    for cossim in cossims:
        cossim_list.append(round(cossim.item(), 5))
    return cossim_list
