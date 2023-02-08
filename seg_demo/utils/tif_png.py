import os
from osgeo import gdal
 
open_path = "C:\\python\\python_pro\\image_cut"
save_path = "C:\\python\\python_pro\\image_cut_png"
 
images = os.listdir(open_path)
for image in images:
    im=gdal.Open(os.path.join(open_path,image))
    driver=gdal.GetDriverByName('PNG')
    dst_ds = driver.CreateCopy(os.path.join(save_path,image.split('.')[0]+".png"), im)