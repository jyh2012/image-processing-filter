import math
import cv2
import numpy as np

def Sin_noise(im):
    '''
        input:
        im 代表输入的二维图像数组
        return:
        result 代表加入椒盐噪声变换之后的结果
    '''
    M, N = im.shape
    sin_im = im.copy()
    for i in range(M):
        for j in range(N):
            sin_noise = np.sin(i) * 20 + np.sin(j) * 20
            sin_im[i][j] += sin_noise
    sin_im = np.where(sin_im < 0, 0, np.where(sin_im > 255, 255, sin_im))
    return sin_im.astype('uint8')


def Gauss_noise(im):
    '''
        input:
        im 代表输入的二维图像数组
        return:
        result 代表加入椒盐噪声变换之后的结果
    '''
    M, N = im.shape
    gauss_noise = np.random.normal(0,20,(M, N))
    gauss_im = im.astype('float') + gauss_noise
    gauss_im = np.where(gauss_im < 0, 0, np.where(gauss_im > 255, 255, gauss_im))
    return gauss_im.astype('uint8')


def Salty_noise(im):
    '''
        input:
        im 代表输入的二维图像数组
        return:
        result 代表加入椒盐噪声变换之后的结果
    '''
    M, N = im.shape
    salt_noise = np.random.randint(0,256,(M, N))
    rand = 0.03
    pepper = np.where(salt_noise < rand * 256, 256, 0)
    noise_im = im.astype('float') + pepper.astype('float')
    noise_im = np.where(noise_im > 255, 255, noise_im)
    return noise_im.astype('uint8')

def main():
    pic1 = cv2.imread(r"house.bmp",0)
    im1 = Salty_noise(pic1)
    im2 = Gauss_noise(pic1)
    im3 = Sin_noise(pic1)
    cv2.imwrite('house_salt.bmp',im1)
    cv2.imwrite('house_gauss.bmp',im2)
    cv2.imwrite('house_sin.bmp',im3)

    pic2 = cv2.imread(r"hill.bmp",0)
    im1 = Salty_noise(pic2)
    im2 = Gauss_noise(pic2)
    im3 = Sin_noise(pic2)
    cv2.imwrite('hill_salt.bmp',im1)
    cv2.imwrite('hill_gauss.bmp',im2)
    cv2.imwrite('hill_sin.bmp',im3)

    pic3 = cv2.imread(r"boat.bmp",0)
    im1 = Salty_noise(pic3)
    im2 = Gauss_noise(pic3)
    im3 = Sin_noise(pic3)
    cv2.imwrite('boat_salt.bmp',im1)
    cv2.imwrite('boat_gauss.bmp',im2)
    cv2.imwrite('boat_sin.bmp',im3)

    pic4 = cv2.imread(r"couple.bmp",0)
    im1 = Salty_noise(pic4)
    im2 = Gauss_noise(pic4)
    im3 = Sin_noise(pic4)
    cv2.imwrite('couple_salt.bmp',im1)
    cv2.imwrite('couple_gauss.bmp',im2)
    cv2.imwrite('couple_sin.bmp',im3)

    pic5 = cv2.imread(r"man.bmp",0)
    im1 = Salty_noise(pic5)
    im2 = Gauss_noise(pic5)
    im3 = Sin_noise(pic5)
    cv2.imwrite('man_salt.bmp',im1)
    cv2.imwrite('man_gauss.bmp',im2)
    cv2.imwrite('man_sin.bmp',im3)


if __name__ == '__main__':
    main()
