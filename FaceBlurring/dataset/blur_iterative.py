import copy
from collections import defaultdict

from tqdm import tqdm
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

def iterative_blur_n(image, n=3, dsize=(112, 112), blur_method_list=['deblurGAN', 'defocus']):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    clean_image = image
    blur_image = copy.deepcopy(image)
    blur_image_list, cossim_list = [], []

    for count in range(1, n+1):
        blur_method = np.random.choice(blur_method_list)
        if blur_method == 'defocus':
            blur_image, blur_image_tensor = defocus_blur_func(blur_image)

        elif blur_method == 'deblurGAN':
            blur_image, blur_image_tensor = deblurGAN_blur_func(blur_image)

        
        cossim = cosine_similarity_func(model, clean_image, blur_image)
        gen_blur_image = cv2.resize(blur_image, dsize=dsize, interpolation=cv2.INTER_AREA)
        blur_image_list.append(gen_blur_image)
        cossim_list.append(cossim)

    clean_image = cv2.resize(clean_image, dsize=dsize, interpolation=cv2.INTER_AREA)
    return clean_image, blur_image_list, cossim_list
  
def defocus_blur_func(img):
    #input image (doesn't matter whether clean or blur)
    #return blurred image when the input is in.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    blur_img, blur_deg = blurring(img, {'mean':50, 'var':0, 'dmin':0, 'dmax':100})
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

def cosine_similarity_func(model, clean_img, blur_img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_img_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    blur_img_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    clean_img_embed = model(clean_img_tensor).squeeze(0).cpu().detach()
    blur_img_embed = model(blur_img_tensor).squeeze(0).cpu().detach()
    cossim = cos_sim(clean_img_embed, blur_img_embed)

    return cossim

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# img = cv2.imread('../data/sample_root/clean/ffhq/69000.png')
# blur_image = copy.deepcopy(img)
# kernel_list = []
# psf_list = []
# blur_method_list=['deblurGAN', 'defocus']
# for i in range(5):
#     blur_method = np.random.choice(blur_method_list)
#     if blur_method == 'defocus':
#         blur_image, blur_image_tensor, kernel = defocus_blur_func(blur_image)
#         kernel_list.append(copy.deepcopy(kernel))

#     elif blur_method == 'deblurGAN':
#         blur_image, blur_image_tensor, psf = deblurGAN_blur_func(blur_image)
#         psf_list.append(psf)
#     print(str(i+1), cosine_similarity_func(model, img, blur_image))
#     cv2.imwrite('../data/blur_'+str(i+1)+'.jpg', blur_image)


# tot_kernel = np.zeros((100, 100))
# for kernel in kernel_list:
#     print('kernel:', kernel.shape)
#     left = (100 - kernel.shape[0]) // 2
#     right = 100 - kernel.shape[0] - left
#     kernel = np.pad(kernel, ((left,right),(left,right)), 'constant', constant_values=0.0)
#     plt.imshow(kernel)
#     plt.show()
#     tot_kernel += kernel
#     plt.imshow(tot_kernel)
#     plt.show()

# tot_psf = np.zeros((64, 64))
# for psf_mini in psf_list:
#     for psf in psf_mini:
#         tot_psf += psf
# plt.imshow(tot_psf)
# plt.show()

# # print(blur_img)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# img_path_list = get_img_path('../data/sample_root/clean')
# cossim_dict = defaultdict(list)
# cossim_list = []

# for img_path in tqdm(img_path_list[:10]):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     for mean in range(0, 101, 10):
#         blur_image, blur_image_tensor = defocus_blur_func(img, mean)
#         cossim = cosine_similarity_func(model, img, blur_image)
#         cossim_dict[mean].append(cossim)
#         print(f'{mean}, {cossim}')

# cossim_list=[]
# cossim_dict = {0: [1.0, 1.0, 1.0, 1.0, 0.99833226, 0.9985055, 1.0, 1.0, 1.0, 1.0], 10: [0.9333347, 0.93840975, 0.98491174, 0.8998525, 0.98541874, 0.97007996, 0.97993606, 0.955158, 0.9421996, 0.97448313], 20: [0.76681113, 0.86881816, 0.78387946, 0.94321483, 0.91493386, 0.8497199, 0.92211896, 0.91200405, 0.73482084, 0.91177833], 30: [0.6391128, 0.88363403, 0.6242507, 0.56252205, 0.8081864, 0.8196665, 0.78995603, 0.79679483, 0.5836229, 0.91704583], 40: [0.48727098, 0.48934722, 0.773649, 0.66794145, 0.8373377, 0.69286716, 0.46941283, 0.72671914, 0.46890804, 0.9225695], 50: [0.72184396, 0.6232076, 0.48361525, 0.59581345, 0.64291894, 0.67487866, 0.3672161, 0.7039092, 0.5385506, 0.6821191], 60: [0.66404945, 0.46501577, 0.56842023, 0.5527489, 0.49894693, 0.6210764, 0.37645793, 0.6586819, 0.3313429, 0.8566661], 70: [0.4910657, 0.3654753, 0.54548234, 0.41559786, 0.45088363, 0.6172431, 0.5277365, 0.6905218, 0.31181327, 0.82241225], 80: [0.3425188, 0.33576864, 0.4198987, 0.5094902, 0.62362415, 0.48028558, 0.5953174, 0.47286654, 0.40363646, 0.48995665], 90: [0.607228, 0.26738703, 0.3568131, 0.4189324, 0.35024834, 0.51094216, 0.252216, 0.5826315, 0.29178908, 0.36547112], 100: [0.39830372, 0.3041494, 0.47212458, 0.4276281, 0.49147302, 0.3912875, 0.24024008, 0.4231802, 0.44278318, 0.62546676]}
# for mean in cossim_dict:
#     cossim_list.append(np.array(cossim_dict[mean]).mean())

# plt.plot(np.arange(0, len(cossim_list))*10, cossim_list)
# plt.xlabel('degree')
# plt.ylabel('cossim')
# plt.show()
    