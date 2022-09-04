import argparse
import pdb

from tqdm import tqdm
import pandas as pd
from blur import *
from utils import *
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn.functional as F
import pickle as pkl
import matplotlib.pyplot as plt
from blur_iterative import cosine_similarity_batch

class CreateBlurImages:
    def __init__(self, data_dir, blur_method='defocus', update=''):
        '''
            class to create blur image dataset from raw images
            Args:
                data_dir : directory of data(clean samples)
                blur_method : blur option(defocus, deblurGAN, random is available)
                update : option update kernels(r, a, or '')
        '''
        # Update filter file(filters.pkl)
        self.update = update
        assert update in ['r', 'a', ''], "Not available update type"
        # Available image files
        self.available = ['.png', '.jpg', 'PNG', 'JPG', 'JPEG']

        # motion blur method
        assert blur_method in ['defocus', 'deblurGAN', 'random']
        self.blur_method = blur_method

        # Get sample paths in list
        self.sample_paths = self._get_all_imgs(data_dir)
        self._create_sample_dirs()

        # padding option to face detection(SCRFD)
        self.pad_max = 200

        # Motion blur hyperparameters
        self.parameters1 = {'mean': 50, 'var': 20, 'dmin': 0, 'dmax': 200}
        self.parameters2 = {'canvas': 64,
                            'iters': 2000,
                            'max_len': 60,
                            'expl': np.random.choice([0.003, 0.001,
                                                      0.0007, 0.0005,
                                                      0.0003, 0.0001]),
                            'part': np.random.choice([1, 2, 3])}

        # Read filter file(filters.pkl)
        if os.path.isfile("filters.pkl"):
            print("Get kernels from file...")
            with open("filters.pkl", "rb") as f:
                self.filters = pkl.load(f)

            if update != '':
                # Generate new filters
                new_filters = self.generate_blur_kernels(False, 100000)

                # Add generated filters to existing dictionary
                if update == 'a':
                    print("Adding kernels...")
                    self.filters['method'] += new_filters['method']
                    self.filters['kernel'] += new_filters['kernel']
                    with open("filters.pkl", 'wb') as f:
                        pkl.dump(self.filters, f, pkl.HIGHEST_PROTOCOL)

                # Replace generated filters with existing dictionary
                elif update == 'r':
                    print("Revising kernels...")
                    self.filters = new_filters
                    with open("filters.pkl", 'wb') as f:
                        pkl.dump(self.filters, f, pkl.HIGHEST_PROTOCOL)

        # Generate filter file if filter file does not exist
        else:
            self.filters = self.generate_blur_kernels(True, 100000)

    def _get_all_imgs(self, root):
        '''
            Function to get all image samples inside the directory
            os.walk will search all directory
            Arg:
                root : data directory of data(clean images)
            ReturnL
                paths : all image paths(clean)
        '''
        paths = []
        print('Check all sample images(clean)...')
        for (path, directory, files) in tqdm(os.walk(root)):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in self.available and 'clean' in path:
                    paths += [os.path.join(path, filename)]
        return paths

    def _create_sample_dirs(self):
        '''
            Function to create sample directories inside the directory
            create directory name with blur_method
        '''
        print('Generate sample directories...')
        for files in tqdm(self.sample_paths):
            path = os.path.dirname(files)
            path2list = path.split(os.path.sep)
            rootpath = os.path.sep.join(path2list[:3])
            subpath = os.path.sep.join(path2list[4:])
            blurpath = os.path.join(rootpath, 'blur_' + self.blur_method, subpath)
            os.makedirs(blurpath, exist_ok=True)

    def generate_blur_kernels(self, save=True, num_filters=100000):
        '''
            Generate random blur kernels
            Save kernels in pkl file
            Args:
                save : option to save kernel files
                num_filters : the number of random filters
            Return:
                generated kernels
        '''
        # Dictionary format(blur method, kernel pair)
        filters = {'method': [], 'kernel': []}
        print("Generating kernels...")
        for _ in tqdm(range(num_filters)):
            blur_method = np.random.choice(['defocus', 'deblurGAN'])

            if blur_method == 'defocus':
                filters['method'].append(blur_method)
                filters['kernel'].append(linear_kernel(self.parameters1))

            elif blur_method == 'deblurGAN':
                filters["method"].append(blur_method)
                self.parameters2['expl'] = np.random.choice([0.003, 0.001,
                                                            0.0007, 0.0005,
                                                            0.0003, 0.0001])
                trajectory = Trajectory(self.parameters2).fit()
                psf = PSF(self.parameters2['canvas'], trajectory=trajectory).fit()
                filters["kernel"].append(psf[np.random.choice([1, 2, 3])])

        # Save option
        if save:
            with open("filters.pkl", 'wb') as f:
                pkl.dump(filters, f, pkl.HIGHEST_PROTOCOL)

        return filters

    def generate_blur_images(self, scrfd=False, num_samples=1, batch=1):
        '''
            Generate blur images with random kernels
            Args:
                scrfd : use scrfd crop and align face image
                num_samples : generate n samples for one clean image
                batch : batch size to create label(recognition inference batch size)
        '''
        # SCRFD normalize and align
        if scrfd:
            app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
        print('Generate blur images...')
        resnet = InceptionResnetV1(pretrained='vggface2', device='cuda' if torch.cuda.is_available() else 'cpu').eval()
        dict_for_label = {'filename': [], 'cosine': []}
        label_batch, clean_img_tensor, blur_img_tensor = 0, [], []
        for image_file in tqdm(self.sample_paths):
            for iter in range(num_samples):
                image = cv2.imread(image_file)
                if image is None:
                    continue

                label_batch += 1
                # Use scrfd to find face
                if scrfd:
                    expanded = image
                    pad = 0
                    find = False
                    while not find and pad <= 200:
                        expanded = np.pad(expanded, ((50, 50), (50, 50), (0, 0)), 'constant', constant_values=0)
                        face_image, find = crop_n_align(app, expanded)
                        pad += 50

                    if find:
                        image = face_image

                # choose random kernel match blur method
                random_idx = np.random.randint(0, len(self.filters['kernel']) - 1)
                if self.blur_method != 'random':
                    while self.filters['method'][random_idx] != self.blur_method:
                        random_idx = np.random.randint(0, len(self.filters['kernel']) - 1)

                blurred = blurring(image, self.filters['kernel'][random_idx])
                path = os.path.dirname(image_file)
                path2list = path.split(os.path.sep)
                rootpath = os.path.sep.join(path2list[:3])
                subpath = os.path.sep.join(path2list[4:])
                blurpath = os.path.join(rootpath, f'blur_{self.blur_method}', subpath)

                assert len(path) + len(self.blur_method) == len(blurpath), 'You should create data directory properly'
                if num_samples > 1:
                    filename = os.path.splitext(os.path.basename(image_file))[0] + f'_{iter}.png'
                elif num_samples == 1:
                    filename = os.path.splitext(os.path.basename(image_file))[0] + '.png'

                cv2.imwrite(os.path.join(blurpath, filename), blurred)
                dict_for_label['filename'] += [os.path.join(blurpath, filename)]

                clean_img_tensor += [torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(
                    'cuda' if torch.cuda.is_available() else 'cpu')]
                blur_img_tensor += [torch.Tensor(blurred).permute(2, 0, 1).unsqueeze(0).to(
                    'cuda' if torch.cuda.is_available() else 'cpu')]

                if label_batch == batch:
                    batch_clean = torch.cat(clean_img_tensor, dim=0)
                    batch_blur = torch.cat(blur_img_tensor, dim=0)
                    emb_clean = resnet(batch_clean)
                    emb_blur = resnet(batch_blur)
                    cosine = F.cosine_similarity(emb_clean, emb_blur)
                    dict_for_label['cosine'] += cosine.tolist()
                    label_batch, clean_img_tensor, blur_img_tensor = 0, [], []

        save_dir = os.path.join('../data', f"label_{self.blur_method}", 'label')
        os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(dict_for_label)
        df.to_csv(os.path.join(save_dir, "label.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program creates blur images.')
    parser.add_argument('--blur', type=str, help='defocus, deblurGAN, random is available', default='random')
    parser.add_argument('--scrfd', type=bool, help='Apply scrfd crop and align on the image', default=False)
    parser.add_argument('--iter', type=int, help="Number of samples to generate blur images", default=1)
    parser.add_argument('--update', type=str, help='update exist filter file(a to add, r to revise)', default='')
    parser.add_argument('--batch', type=int, help="Batch size of labeling", default=1)
    args = parser.parse_args()

    blurrer = CreateBlurImages("../data", args.blur, args.update)
    blurrer.generate_blur_images(args.scrfd, args.iter, args.batch)


