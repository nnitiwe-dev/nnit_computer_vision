#import libraries
import numpy as np
import cv2
from skimage import exposure


class Noise:
	def add_noise(noise_typ,image):
		'''

		:param noise_typ: Category of noise to apply on image.
		:type noise_typ: str
		:param image: Image Data as numpy array
        :type image: np.array

        :returns: Image (np.array)

		'''
		if noise_typ == "gaussian":
			row,col,ch= image.shape
			mean = 0
			var = 0.1
			sigma = var**0.5
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			noisy = image + gauss
			return noisy

		elif noise_typ == "salt_pepper":
		    row,col,ch = image.shape
		    s_vs_p = 0.5
		    amount = 0.004
		    out = np.copy(image)

		    # Salt mode
		    num_salt = np.ceil(amount * image.size * s_vs_p)
		    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
		    out[coords] = 1

		    # Pepper mode
		    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		    out[coords] = 0
		    return out

		elif noise_typ == "poisson":
		    vals = len(np.unique(image))
		    vals = 2 ** np.ceil(np.log2(vals))
		    noisy = np.random.poisson(image * vals) / float(vals)
		    return noisy

		elif noise_typ =="speckle":
		    row,col,ch = image.shape
		    gauss = np.random.randn(row,col,ch)
		    gauss = gauss.reshape(row,col,ch)
		    noisy = image + image * gauss
		    return noisy

		elif noise_typ =="gamma":
		    img = exposure.adjust_gamma(image.astype(np.uint8), 7)
		    noisy = img 
		    return noisy

		elif noise_typ =="log":
		    img = exposure.adjust_log(image.astype(np.uint8), 10)
		    noisy = img 
		    return noisy

#usage
if __name__ == '__main__':
	
	#Load image
	image=cv2.imread('PATH_TO_IMAGE')

	Noisy=Noise()
	noisy_img=Noisy.add_noise('salt_pepper',image)



