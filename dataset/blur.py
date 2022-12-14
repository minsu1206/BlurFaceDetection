import random
from utils import *

def linear_kernel(param, random_method='uniform'):
    '''
        Create linear kernel(defocus method)
        input : param, random_method
        output = blur kernel
    '''
    mean, var, dmin, dmax = param['mean'], param['var'], param['dmin'], param['dmax']
    if random_method == 'gaussian':
        random_degree = dmax + 1
        while random_degree < dmin or random_degree > dmax:
            random_degree = int(random.normalvariate(mean, var))
    elif random_method == 'uniform':
        random_degree = random.randint(dmin, dmax)
    else:
        raise ValueError("This metric is not available(choose from uniform, gaussian)")

    if random_degree == 1 or random_degree == 0:
        random_angle = random.randint(-88, 88)
        random_degree = 1
    else:
        random_angle = random.randint(-180, 180)
    # Create random motion blur kernel
    M = cv2.getRotationMatrix2D((random_degree / 2, random_degree / 2), random_angle, 1)
    kernel = np.diag(np.ones(random_degree))
    kernel = cv2.warpAffine(kernel, M, (random_degree, random_degree))
    kernel = kernel / random_degree

    return kernel

def blurring(img, kernel=None):
    '''
        Apply motion blur to the image(Both defocus and deblurGAN method)

    '''
    if kernel is None:
        choice = np.random.randint(2)
        if choice == 0:
            kernel = linear_kernel(param={'mean': 50, 'var': 20, 'dmin': 0, 'dmax': 200})
        else:
            trajectory = Trajectory({'canvas':64, 'iters':2000, 'max_len':60,
                                     'expl': np.random.choice([0.003, 0.001,
                                                               0.0007, 0.0005,
                                                               0.0003, 0.0001])}).fit()
            psf = PSF(64, trajectory=trajectory).fit()
            kernel = psf[np.random.choice([1, 2, 3])]
    # Apply kernel on the image sample
    image = np.array(img)
    blurred = cv2.filter2D(image, -1, kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


class Trajectory(object):
    def __init__(self, param):
        self.canvas = param['canvas']
        self.iters = param['iters']
        self.max_len = param['max_len']

        if param['expl'] is None:
            self.expl = 0.1 * np.random.uniform(0, 1)
        else:
            self.expl = param['expl']

        self.tot_length = None
        self.big_expl_count = None
        self.x = None

    def fit(self):
        tot_length = 0
        big_expl_count = 0
        centripetal = 0.7 * np.random.uniform(0, 1)
        prob_big_shake = 0.2 * np.random.uniform(0, 1)
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle = 360 * np.random.uniform(0, 1)
        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl

        x = np.array([complex(real=0, imag=0)] * (self.iters))

        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=math.ceil((self.canvas - max(x.real)) / 2), imag=math.ceil((self.canvas - max(x.imag)) / 2))

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x

        return self

class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False)
        else:
            self.trajectory = trajectory.x

        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self):
        PSF = np.zeros(self.canvas)
        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))

        trajectory_mag = 0.0
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )
            self.PSFs.append(PSF / (self.iters))
        return self.PSFs
