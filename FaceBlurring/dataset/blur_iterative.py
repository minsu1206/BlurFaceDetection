from blur import *
from facenet_pytorch import MTCNN, InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

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
    
def defocus_blur_func(img):
    #input image (doesn't matter whether clean or blur)
    #return blurred image when the input is in.
    blur_img, blur_deg = blurring(img, {'mean':50, 'var':20, 'dmin':0, 'dmax':10})
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).cuda()
    return blur_img, blur_img_tensor


def deblurGAN_blur_func(img):
    trajectory = Trajectory({'canvas':64, 'iters':2000, 'max_len':60, 'expl':0.001}).fit()
    psf, degree = PSF(64, trajectory).fit()
    part_parameters = np.random.choice([1, 2, 3])
    clean_img, blur_img = BlurringImage(img, psf, part_parameters).blur_image()
    blur_img_tensor = torch.Tensor(blur_img).permute(2, 0, 1).unsqueeze(0).cuda()
    return blur_img, blur_img_tensor


def cosine_similarity_func(clean_img, blur_img):
    clean_img_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
    blur_img_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
    clean_img_embed = resnet(clean_img_tensor).squeeze(0).cpu().detach()
    blur_img_embed = resnet(blur_img_tensor).squeeze(0).cpu().detach()
    cossim = cos_sim(clean_img_embed, blur_img_embed)

    return cossim