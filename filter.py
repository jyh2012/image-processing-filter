# coding=utf-8
import cv2
import numpy as np
from scipy.signal import butter,lfilter
'''
opencv 理想滤波、巴特沃兹滤波和高斯滤波的高通、低通滤波演示
Filter Parameters 窗口： 滤波器参数窗口
- d0: 滤波器大小 D0
- flag: 滤波器类型
0 - 理想滤波
1 - 巴特沃兹滤波
2 - 高斯滤波
- n: 巴特沃兹滤波的阶数
- lh: 低通滤波 or 高通滤波
Filtered Image 窗口: 过滤后的图像和过滤器图像
'''
 
 
def combine_images(images, axis=1):
    '''
    合并图像。
    @param images: 图像列表(图像成员的维数必须相同)
    @param axis: 合并方向。 
    axis=0时，图像垂直合并;
    axis = 1 时， 图像水平合并。
    @return 合并后的图像
    '''
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
    if axis == 0:  # 垂直方向合并图像
        # 合并图像的 cols
        cols = np.max(shapes[:, 1])
        # 扩展各图像 cols大小，使得 cols一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 垂直方向合并
        return np.vstack(copy_imgs)
    else:  # 水平方向合并图像
        # 合并图像的 rows
        rows = np.max(shapes[:, 0])
        # 扩展各图像rows大小，使得 rows一致
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        # 水平方向合并
        return np.hstack(copy_imgs)
 
 
def fft(img):
    '''对图像进行傅立叶变换，并返回换位后的频率矩阵'''
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
    # 计算最优尺寸
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    # 根据新尺寸，建立新变换图像
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img
    # 傅立叶变换
    fft_mat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 换位，低频部分移到中间，高频部分移到四周
    return np.fft.fftshift(fft_mat)
 
 
def fft_image(fft_mat):
    '''将频率矩阵转换为可视图像'''
    # log函数中加1，避免log(0)出现.
    log_mat = cv2.log(1 + cv2.magnitude(fft_mat[:, :, 0], fft_mat[:, :, 1]))
    # 标准化到0~255之间
    cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(log_mat))
 
 
def ifft(fft_mat):
    '''傅立叶反变换，返回反变换图像'''
    # 反换位，低频部分移到四周，高频部分移到中间
    f_ishift_mat = np.fft.ifftshift(fft_mat)
    # 傅立叶反变换
    img_back = cv2.idft(f_ishift_mat)
    # 将复数转换为幅度, sqrt(re^2 + im^2)
    img_back = cv2.magnitude(*cv2.split(img_back))
    # 标准化到0~255之间
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.around(img_back))
 
 
def fft_distances(m, n):
    '''
    计算m,n矩阵每一点距离中心的距离
    见《数字图像处理MATLAB版.冈萨雷斯》93页
    '''
    u = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v = np.array([i if i <= m / 2 else m - i for i in range(m)],
                 dtype=np.float32)
    v.shape = n, 1
    # 每点距离矩阵左上角的距离
    ret = np.sqrt(u * u + v * v)
    # 每点距离矩阵中心的距离
    return np.fft.fftshift(ret)
 
 
def lpfilter(flag, rows, cols, d0, n):
    '''低通滤波器
    @param flag: 滤波器类型
    0 - 理想低通滤波
    1 - 巴特沃兹低通滤波
    2 - 高斯低通滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹低通滤波的阶数 
    @return 滤波器矩阵 
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想低通滤波
    if flag == 0:
        filter_mat = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   d0, (1, 1, 1), thickness=-1)
    # 巴特沃兹低通滤波
    elif flag == 1:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 / (1 + np.power(duv / d0, 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯低通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat
 
 
def hpfilter(flag, rows, cols, d0, n):
    '''高通滤波器
    @param flag: 滤波器类型
    0 - 理想高通滤波
    1 - 巴特沃兹高通滤波
    2 - 高斯高通滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹高通滤波的阶数 
    @return 滤波器矩阵 
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想高通滤波
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   d0, (0, 0, 0), thickness=-1)
    # 巴特沃兹高通滤波
    elif flag == 1:
        duv = fft_distances(rows, cols)
    # duv有 0 值(中心距离中心为0)， 为避免除以0，设中心为 0.000001
        duv[rows // 2, cols // 2] = 0.000001
        filter_mat = 1 / (1 + np.power(d0 / duv, 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯高通滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-(duv * duv) / (2 * d0 * d0))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat


def npfilter(flag, rows, cols, d0, n):
    '''陷波滤波器
    @param flag: 滤波器类型
    0 - 理想陷波滤波
    1 - 巴特沃兹陷波滤波
    2 - 高斯陷波滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param d0: 滤波器大小 D0
    @param n: 巴特沃兹高通滤波的阶数 
    @return 滤波器矩阵 
    '''
    assert d0 > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想陷波滤波
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 4, cols // 2),
                    d0 , (0, 0, 0), thickness=-1)
        
        cv2.circle(filter_mat, (rows * 3 // 4, cols // 2),
                    d0 , (0, 0, 0), thickness=-1)
    elif flag == 1:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 4),
                    d0 , (0, 0, 0), thickness=-1)
        
        cv2.circle(filter_mat, (rows // 2, cols * 3 // 4),
                    d0 , (0, 0, 0), thickness=-1)

    return filter_mat


def bspfilter(flag, rows, cols, lowband, highband, n):
    '''带阻滤波器
    @param flag: 滤波器类型
    0 - 理想带阻滤波
    1 - 巴特沃兹带阻滤波
    2 - 高斯带阻滤波
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param lowband: 滤波器内半径大小 Lowband
    @param highband：滤波器外半径大小 Highband
    @param n: 巴特沃兹高通滤波的阶数 
    @return 滤波器矩阵 
    '''
    assert lowband > 0, 'd0 should be more than 0.'
    filter_mat = None
    # 理想带阻滤波
    if flag == 0:
        filter_mat = np.ones((rows, cols, 2), np.float32)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   highband, (0, 0, 0), thickness=-1)
        cv2.circle(filter_mat, (rows // 2, cols // 2),
                   lowband, (1, 1, 1), thickness=-1)
    # 巴特沃兹带阻滤波
    elif flag == 1:
        duv = fft_distances(rows, cols)
    # duv有 0 值(中心距离中心为0)， 为避免除以0，设中心为 0.000001
        duv[rows // 2, cols // 2] = 0.000001
        filter_mat = 1 / (1 + np.power(duv * (highband - lowband) / (duv*duv - lowband*lowband), 2 * n))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    # 高斯带阻滤波
    else:
        duv = fft_distances(*fft_mat.shape[:2])
        filter_mat = 1 - np.exp(-np.power((duv * duv - lowband * lowband) / (duv * (highband - lowband)), 2))
        # fft_mat有2个通道，实部和虚部
        # fliter_mat 也需要2个通道
        filter_mat = cv2.merge((filter_mat, filter_mat))
    return filter_mat

def do_filter(_=None):
    '''滤波，并显示'''
    d0 = cv2.getTrackbarPos('d0', filter_win)
    flag = cv2.getTrackbarPos('flag', filter_win)
    n = cv2.getTrackbarPos('n', filter_win)
    lh = cv2.getTrackbarPos('lh', filter_win)
    lowband = cv2.getTrackbarPos('lowband', filter_win)
    highband = cv2.getTrackbarPos('highband', filter_win)
    # 滤波器
    filter_mat = None
    if lh == 0:
        filter_mat = lpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    elif lh == 1:
        filter_mat = hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    elif lh == 2:
        filter_mat = npfilter(0, fft_mat.shape[0], fft_mat.shape[1], d0, n) * npfilter(1, fft_mat.shape[0], fft_mat.shape[1], d0, n)
    elif lh == 3:
        filter_mat = bspfilter(flag, fft_mat.shape[0], fft_mat.shape[1], lowband, highband, n)
        #filter_mat = hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], highband, n) - hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], lowband, n)
    elif lh == 4:
        filter_mat = hpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], highband, n) + lpfilter(flag, fft_mat.shape[0], fft_mat.shape[1], lowband, n)
        
    # 进行滤波
    filtered_mat = filter_mat * fft_mat
    # 反变换
    img_back = ifft(filtered_mat)
    # 显示滤波后的图像和滤波器图像
    back_mat = fft(img_back)
    combine_1 = combine_images([img, fft_image(fft_mat), fft_image(filter_mat)])
    combine_2 = combine_images([img_back, fft_image(back_mat), fft_image(filtered_mat)])
    cv2.imshow(image_win, combine_images([combine_1, combine_2], 0))



if __name__ == '__main__':
    img = cv2.imread(r'house_sin.bmp', 0)
    rows, cols = img.shape[:2]
    # 滤波器窗口名称
    filter_win = 'Filter Parameters'
    # 图像窗口名称
    image_win = 'Filtered Image'
    cv2.namedWindow(filter_win)
    cv2.namedWindow(image_win)
    # 创建d0 tracker, d0为过滤器大小
    cv2.createTrackbar('d0', filter_win, 20, min(rows, cols) // 4, do_filter)
    # 创建flag tracker,
    # flag=0时，为理想滤波
    # flag=1时，为巴特沃兹滤波
    # flag=2时，为高斯滤波
    cv2.createTrackbar('flag', filter_win, 1, 2, do_filter)
    # 创建n tracker
    # n 为巴特沃兹滤波的阶数
    cv2.createTrackbar('n', filter_win, 5, 10, do_filter)
    # 创建lh tracker
    # lh: 滤波器是低通还是高通， 0 为低通， 1为高通， 2为陷波， 3为带阻1， 4为带阻2
    cv2.createTrackbar('lh', filter_win, 0, 4, do_filter)
    # 创建lowband tracker
    # lowband:带通、带阻滤波器的低频下限
    cv2.createTrackbar('lowband', filter_win, 70, 200, do_filter)
    # 创建highband tracker
    # highband:带通、带阻滤波器的高频上限
    cv2.createTrackbar('highband', filter_win, 100, 200, do_filter)
    fft_mat = fft(img)
    do_filter()
    cv2.resizeWindow(filter_win, 512, 50)
    cv2.waitKey(0)
    cv2.destroyAllWindows()