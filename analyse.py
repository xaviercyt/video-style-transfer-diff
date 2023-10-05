from skimage.metrics import mean_squared_error
from skimage import io
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim


# 读取原始图像和对比图像
original_image = Image.open('./image/未命名文件夹 3/231693608828_.pic_hd.jpg')
comparison_image = Image.open('./image/未命名文件夹 3/241693608844_.pic_hd.jpg')
# original_image = Image.open('./Imagesnew/longhair/sample2.jpeg')
# comparison_image = Image.open('./Imagesnew/311693672634_.pic_thumb.jpg')

# 调整图像尺寸为相同维度
width, height = min(original_image.width, comparison_image.width), min(original_image.height, comparison_image.height)
original_image = original_image.resize((width, height))
comparison_image = comparison_image.resize((width, height))

# 将图像转换为 numpy 数组
original_array = np.array(original_image)
comparison_array = np.array(comparison_image)

# 计算均方误差
mse = mean_squared_error(original_array, comparison_array)
print(f"MSE: {mse}")

ssim_value = ssim(original_array, comparison_array, multichannel=True)
print(f"SSIM: {ssim_value}")

#%%
