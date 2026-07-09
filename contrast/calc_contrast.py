from PIL import Image
import numpy as np

conts = []
sharps = []
thzs = np.arange(0.2, 3.1, 0.1)
for thz in thzs:
    img = Image.open(f'../thz_new/resol1951_2_e_{round(thz, 1)}THz.png')
    img = np.asarray(img)
    
    # calculate the gradient
    gy, gx = np.gradient(img)
    g_norm = np.sqrt(gx**2 + gy**2)

    # compute the sharpness value
    sharpness = np.average(g_norm)
    sharps.append(sharpness)
    
    contrast = img.std()
    print(round(thz, 1), contrast, sharpness)
    conts.append(contrast)

print('MIN Contrast:', round(thzs[np.argmin(conts)], 1), min(conts))    
print('MAX Contrast:', round(thzs[np.argmax(conts)], 1), max(conts))

print('MIN Shaprness:', round(thzs[np.argmin(sharps)], 1), min(sharps))    
print('MAX Sharpness:', round(thzs[np.argmax(sharps)], 1), max(sharps))

