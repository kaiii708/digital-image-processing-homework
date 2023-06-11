import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# def global_histogram_equlization(img):
#     min = 0
#     max = 255
#     minval,maxval,minloc,maxloc = cv2.minMaxLoc(img)
#     average = img[:,:].sum() //(height*width)
#     turn_point = round(average/4)
#     slope1 = round(max*2/3/(turn_point-minval))
#     slope2 = round(max/3/(maxval-turn_point))
#     new_img = np.zeros(img.shape)
#     for i in range(0,img.shape[0]):
#         for j in range(img.shape[1]):
#             pixval = img[i][j]
#             if pixval <= turn_point:
#                 new_pixval = (pixval - minval)*slope1
#             else:
#                 new_pixval = round(max*2/3+(pixval - turn_point)*slope2)
#             new_img[i][j] = new_pixval
#     return new_img

def GHE(img):
    min = 0
    max = 255
    height,width = img.shape
    hist,bin_edges = np.histogram(img,bins = 256,range = (min,max))
    print(f"hist:{hist}")
    PDF = hist/(height*width)
    CDF = PDF.cumsum()
    table = np.around(CDF*256)
    print(f"CDF:{CDF}")
    new_img = np.zeros(img.shape)
    for i in range(0,height):
        for j in range(0,width):
            index = round(img[i][j])
            index = 255 if index >255 else index
            new_pixval = table[index]
            new_img[i][j] = new_pixval
    return new_img

def LHE(img,winsize):
    min = 0
    max = 255
    height,width = img.shape
    if winsize%2 == 0:winsize+=1
    half_winsize=(winsize-1)//2
    pad_img = cv2.copyMakeBorder(img,half_winsize,half_winsize,half_winsize,half_winsize,cv2.BORDER_REFLECT)
    # cv2.imshow('Image', pad_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    new_img = np.zeros(pad_img.shape)
    for i in range(half_winsize,height+half_winsize):
        for j in range(half_winsize,width+half_winsize):
            window = pad_img[i-half_winsize:i+half_winsize+1,j-half_winsize:j+half_winsize+1]
            hist,bin_edges = np.histogram(window,bins = 256,range = (min,max))
            PDF = hist/window.size
            CDF = PDF.cumsum()
            table = np.around(CDF*256)
            index = round(pad_img[i][j])
            index = 255 if index >255 else index
            new_img[i][j] = table[index]
    new_img = new_img[half_winsize:half_winsize+height,half_winsize:half_winsize+width]
    return new_img



sample1 = cv2.imread('./hw1_sample_images/sample1.png')
sample2 = cv2.imread('./hw1_sample_images/sample2.png',0)
sample3 = cv2.imread('./hw1_sample_images/sample3.png',0)
sample4 = cv2.imread('./hw1_sample_images/sample4.png',0)
sample5 = cv2.imread('./hw1_sample_images/sample5.png',0)

# cv2.imshow('Image', sample1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# p0(a)
sample1 = cv2.cvtColor(sample1, cv2.COLOR_BGR2RGB)
# BGR
result1 = 0.114*sample1[:,:,0]+0.587*sample1[:,:,1]+0.299*sample1[:,:,2]

# img_gray = cv2.cvtColor(sample1, cv2.COLOR_BGR2GRAY)

cv2.imwrite("result1.png",result1)


# p0(b)
result2 = result1.copy()
height,width = result2.shape
for j in range(0,width):
    for i in range(0,height//2):
        result2[i][j],result2[height-1-i][j] = result2[height-1-i][j],result2[i][j]
cv2.imwrite("result2.png",result2)

# p1(a)
result3 = sample2/3
result3 = np.around(result3)

cv2.imwrite("result3.png",result3)

# p1(b)
result4 = (result3+1)*3
cv2.imwrite("result4.png",result4)

def plot_histogram(img,return_name):
    plt.hist(img,bins=[0,51,102,153,204,255])
    plt.title(return_name)
    # plt.show()
    plt.savefig(return_name)
    plt.clf()

plot_histogram(sample2,"sample2_his")
plot_histogram(result3,"result3_his")
plot_histogram(result4,"result4_his")



# p1(d)
result5 = GHE(sample2)
cv2.imwrite("result5.png",result5)

result6 = GHE(result3)
cv2.imwrite("result6.png",result6)

result7 = GHE(result4)
cv2.imwrite("result7.png",result7)


# p1(e)
result8 = LHE(sample2,200)
cv2.imwrite("result8.png",result8)
result9 = LHE(result3,200)
cv2.imwrite("result9.png",result9)
result10 = LHE(result4,200)
cv2.imwrite("result10.png",result10)

plot_histogram(result5,"result5_his")
plot_histogram(result6,"result6_his")
plot_histogram(result7,"result7_his")
plot_histogram(result8,"result8_his")
plot_histogram(result9,"result9_his")
plot_histogram(result10,"result10_his")

# p1(f)
result11 = LHE(sample2,300)
cv2.imwrite("result11.png",result11)

plot_histogram(result11,"result11_his")

# p2(a)
def low_pass_filter(img,winsize):
    height,width = img.shape
    center = winsize // 2
    mask = np.ones((winsize,winsize))/(winsize*winsize)
    pad_img = cv2.copyMakeBorder(img,center,center,center,center,cv2.BORDER_REFLECT)
    new_img = np.zeros(pad_img.shape)
    for i in range(center,center+height):
        for j in range(center,center+width):
            g = mask[-1,-1]*pad_img[i-1,j-1]+mask[-1,0]*pad_img[i-1,j]+mask[-1,1]*pad_img[i-1,j+1]+mask[0,-1]*pad_img[i,j-1]+mask[0,0]*pad_img[i,j]+mask[0,1]*pad_img[i,j+1]+mask[1,-1]*pad_img[i+1,j-1]+mask[1,0]*pad_img[i+1,j]+mask[1,1]*pad_img[i+1,j+1]
            new_img[i,j] = g
    new_img = new_img[center:center+height,center:center+width]
    new_img = np.around(new_img)
    return new_img

def low_pass_filter_1(img,b):
    winsize = 3
    height,width = img.shape
    center = winsize // 2
    mask = np.array([[1,b,1],[b,pow(b,2),b],[1,b,1]])/pow(b+2,2)
    pad_img = cv2.copyMakeBorder(img,center,center,center,center,cv2.BORDER_REFLECT)
    new_img = np.zeros(pad_img.shape)
    for i in range(center,center+height):
        for j in range(center,center+width):
            g = mask[-1,-1]*pad_img[i-1,j-1]+mask[-1,0]*pad_img[i-1,j]+mask[-1,1]*pad_img[i-1,j+1]+mask[0,-1]*pad_img[i,j-1]+mask[0,0]*pad_img[i,j]+mask[0,1]*pad_img[i,j+1]+mask[1,-1]*pad_img[i+1,j-1]+mask[1,0]*pad_img[i+1,j]+mask[1,1]*pad_img[i+1,j+1]
            new_img[i,j] = g
    new_img = new_img[center:center+height,center:center+width]
    new_img = np.around(new_img)
    return new_img


result12 = low_pass_filter(sample4,3)
cv2.imwrite("result12.png",result12)

# result12_0 = low_pass_filter(sample4,5)

# result12_2 = result12_0 * 1.2
# cv2.imwrite("result12_2.png",result12_2)

# result12_1 = low_pass_filter_1(sample4,2)
# cv2.imwrite("result12_1.png",result12_1)

# result12_3 = low_pass_filter_1(sample4,3)
# cv2.imwrite("result12_3.png",result12_3)

def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2) 
    return 10 * math.log(math.pow(255, 2) / mse, 10)

print(f"psnr between sample3 and result12{psnr(sample3,result12)}")
# print(f"psnr between sample3 and result12_2{psnr(sample3,result12_2)}")
# print(f"psnr between sample3 and result12_1{psnr(sample3,result12_1)}")

def median_filter(img,winsize):
    height,width = img.shape
    half_winsize = winsize //2
    pad_img = cv2.copyMakeBorder(img,half_winsize,half_winsize,half_winsize,half_winsize,cv2.BORDER_REFLECT)
    new_img = np.zeros(pad_img.shape)
    for i in range(half_winsize,half_winsize+height):
        for j in range(half_winsize,half_winsize+width):
            new_img[i,j]=np.median(pad_img[i-half_winsize:i+half_winsize+1,j-half_winsize:j+half_winsize+1])
    new_img=new_img[half_winsize:half_winsize+height,half_winsize:half_winsize+width]
    return new_img

result13 = median_filter(sample5,3)
cv2.imwrite("result13.png",result13)

# print(psnr(sample3,result12))
print(f"psnr between sample3 and result13{psnr(sample3,result13)}")

# print(f"psnr between sample3 and result12_3{psnr(sample3,result12_3)}")