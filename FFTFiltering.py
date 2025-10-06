# Import libraries
import os
import matplotlib.pyplot as plt
import imageio.v3 as io
import skimage.color as color
import numpy as np  
from scipy import ndimage
from scipy.signal import convolve2d
path =r'C:\Users\danie\Desktop\IPCV\Foto Esercitazioni'

Img=io.imread(os.path.join(path,'lighthouse.png'))

Img1=Img.copy()
Img1=Img1.astype(np.float32)/255
if len(Img1.shape)==3:
    Img1gray= color.rgb2gray(Img1)

Ifft=np.fft.fft2(Img1gray)

Ifft=np.fft.fftshift(Ifft)


plt.figure(figsize=(12, 6))
plt.subplot(1,2,1) 
plt.imshow(Img1gray, cmap ='gray'), plt.title('Original')
plt.subplot(1,2,2) 
plt.imshow(np.log(np.abs(Ifft)), cmap ='bwr'), plt.title('Fourier Transform')
plt.show()