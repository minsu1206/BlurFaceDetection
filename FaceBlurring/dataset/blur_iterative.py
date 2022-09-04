import copy
import torch
import torch.nn.functional as F
from blur import *
import pickle as pkl

def iterative_blur_n(model, image, n=3, dsize=(112, 112), blur_method_list=['deblurGAN', 'defocus'], device='cpu', kernel=False):
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
    # [9/4 수정] -> Apply revised code
    # (1) redundant model build -> remove!
    # (2) embedding -> batch forwarding!

    clean_image = image
    clean_img_tensor = torch.from_numpy(clean_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    blur_image = copy.deepcopy(image)
    image_tensor_list, gen_image_list, cossim_list = [clean_img_tensor], [], []

    if kernel:
        try:
            with open("filters.pkl", "rb") as f:
                filters = pkl.load(f)
        except:
            print("Filter file is not exist")
            filters = None
    else:
        filters = None


    for count in range(1, n + 1):
        blur_method = np.random.choice(blur_method_list)
        if blur_method == 'defocus':
            if filters is None:
                blur_image, blur_image_tensor = defocus_blur_func(blur_image)
            else:
                random_idx = np.random.randint(0, len(filters['kernel']) - 1)
                while filters['method'][random_idx] != blur_method:
                    random_idx = np.random.randint(0, len(filters['kernel']) - 1)
                blur_image, blur_image_tensor = defocus_blur_func(blur_image, filters['kernel'][random_idx])

        elif blur_method == 'deblurGAN':
            if filters is None:
                blur_image, blur_image_tensor = defocus_blur_func(blur_image)
            else:
                random_idx = np.random.randint(0, len(filters['kernel']) - 1)
                while filters['method'][random_idx] != blur_method:
                    random_idx = np.random.randint(0, len(filters['kernel']) - 1)
                blur_image, blur_image_tensor = deblurGAN_blur_func(blur_image, filters['kernel'][random_idx])

        gen_blur_image = cv2.resize(blur_image, dsize=dsize, interpolation=cv2.INTER_AREA)
        image_tensor_list.append(blur_image_tensor)
        gen_image_list.append(gen_blur_image)

    batch_imgs = torch.cat(image_tensor_list, dim=0)
    cossim_list = cosine_similarity_batch(model, batch_imgs)
    print(cossim_list)
    clean_image = cv2.resize(clean_image, dsize=dsize, interpolation=cv2.INTER_AREA)
    return clean_image, gen_image_list, cossim_list


def defocus_blur_func(img, kernel=None):
    # input image (doesn't matter whether clean or blur)
    # return blurred image when the input is in.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if kernel is None:
        kernel = linear_kernel({'mean': 25, 'var': 10, 'dmin': 0, 'dmax': 50})

    blur_img = blurring(img, kernel)
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).to(device)

    return blur_img, blur_img_tensor


def deblurGAN_blur_func(img, kernel=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if kernel is None:
        trajectory = Trajectory({'canvas': 64, 'iters': 2000, 'max_len': 60, 'expl': 0.001}).fit()
        kernel = PSF(64, trajectory).fit()[np.random.choice([1, 2, 3])]

    blur_img = blurring(img, kernel)
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).to(device)
    return blur_img, blur_img_tensor


def cosine_similarity_batch(model, imgs):
    img_embed = model(imgs)
    clear_img_embed = img_embed[0, :]  # (F)
    blur_imgs_embed = img_embed[1:, :]  # (B-1, F)

    cossim_list = []
    # Batch Cosine Similarity (Not on CPU. CUDA is faster. Detach follows CUDA matrix operation)

    cossims = F.cosine_similarity(clear_img_embed, blur_imgs_embed)
    cossims = cossims.cpu().detach()
    for cossim in cossims:
        cossim_list.append(round(cossim.item(), 5))
    return cossim_list