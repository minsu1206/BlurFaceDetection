import matplotlib.pyplot as plt
import numpy as np
import cv2
'''
im1 = cv2.imread('./fix_L1L2.png')
im2 = cv2.imread('./fix_COS.png')

im1 = im1[80:700-30,100:1200-100,:]
im2 = im2[60:700-50,100:1200-100,:]
comp = np.concatenate((im1, im2), 0)
cv2.imwrite("./fix_comp.png", comp)
'''

plt.figure(figsize=(12,7))
plt.plot(np.arange(100), label='test')
plt.xlabel(fontsize=15)
plt.legend(fontsize=18)
plt.show()
