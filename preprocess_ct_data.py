import SimpleITK as sitk
import os
import uuid
import shutil
import numpy as np
import nibabel as nib
import zipfile
import random
from PIL import Image
from tqdm import tqdm

def convert_to_float32(image):
    """
    Convert image to float32 if it's not already of this type.
    """
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def resample_image(image):
    new_size = [512, 512, 1]
    new_spacing = [1.0, 1.0, 1.0]  # 目标间距
    new_origin = [0.0, 0.0, 0.0]  # 固定原点
    new_direction = [1.0, 0.0, 0.0,  # 固定方向矩阵
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]
    image = convert_to_float32(image)  # Convert to float32
    """
    Resample the given image to the new size, spacing, origin, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(new_direction)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def dicom_series_to_mha(folder):
    reader = sitk.ImageFileReader()
    resampled_images = []

    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.lower().endswith('.dcm'):
                file_path = os.path.join(root, file)
                reader.SetFileName(file_path)
                image = reader.Execute()
                # Ensure the image is 3D
                if image.GetDimension() == 3:
                    resampled_image = resample_image(image)
                    resampled_images.append(resampled_image)

    # 合并所有重新采样的图像
    if len(resampled_images) > 1:
        final_image = sitk.JoinSeries(resampled_images)
    else:
        final_image = resampled_images[0]

    random_uuid = uuid.uuid4()
    output_mha_file = 'CT/' + str(random_uuid) + '.mha'
    # 重新采样图像
    resampled_image = resample_image(final_image)
    sitk.WriteImage(resampled_image, output_mha_file)
    print(f"Saved MHA file to {output_mha_file}")

def nii_2_mha(folder):
    # [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk(folder) for d in dirs if d == 'segmentations']
    files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if
             file.endswith('.nii.gz')]
    for nii_file in files:
        try:
            # 读取 NIfTI 文件
            sitk_img = sitk.ReadImage(nii_file)
            # nii_img = nib.load(nii_file)
            #
            # # 获取图像数据、空间分辨率和方向
            # img_data = nii_img.get_fdata()
            # spacing = [1.5, 1.5, 1.5]  # 目标间距
            # # origin = [0.0, 0.0, 0.0]  # 固定原点
            # # direction = [0.0, 0.0, 1.0,  # 固定方向矩阵
            # #                 0.0, 1.0, 0.0,
            # #                 1.0, 0.0, 0.0]
            # # spacing = nii_img.header.get_zooms()[:3]  # 只取前三个轴的 spacing
            # # print(spacing)
            # origin = nii_img.affine[:3, 3]
            # direction = nii_img.affine[:3, :3].flatten().tolist()  # 转换为列表格式
            # # print(direction)
            #
            # # 创建 SimpleITK 图像
            # sitk_img = sitk.GetImageFromArray(img_data)
            # sitk_img.SetSpacing(tuple(spacing))  # 将 spacing 转换为元组
            # sitk_img.SetOrigin(tuple(origin))  # 将 origin 转换为元组
            # sitk_img.SetDirection(direction)  # direction 已经是列表格式
            # # 定义目标图像大小、空间分辨率和方向
            # size = nii_image.GetSize()
            # spacing = nii_image.GetSpacing()
            # direction = nii_image.GetDirection()
            #
            # # 创建一个新的图像并重新采样
            # resampled_image = sitk.Resample(nii_image, size, sitk.Transform(), sitk.sitkLinear, nii_image.GetOrigin(),
            #                                 spacing, sitk.sitkIdentity)
            random_uuid = uuid.uuid4()
            output_mha_file = 'CT/' + str(random_uuid) + '.mha'
            sitk.WriteImage(sitk_img, output_mha_file)
        except:
            pass

def nrrd_2_mha(folder):
    # [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk(folder) for d in dirs if d == 'segmentations']
    files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if
             file.endswith('.nrrd')]
    for nrrd_file in files:
        try:
            # 读取 NIfTI 文件
            sitk_img = sitk.ReadImage(nrrd_file)
            # nrrd_img = nib.load(nrrd_file)
            #
            # # 获取图像数据、空间分辨率和方向
            # img_data = nrrd_img.get_fdata()
            # spacing = [1.5, 1.5, 1.5]  # 目标间距
            # # origin = [0.0, 0.0, 0.0]  # 固定原点
            # # direction = [0.0, 0.0, 1.0,  # 固定方向矩阵
            # #                 0.0, 1.0, 0.0,
            # #                 1.0, 0.0, 0.0]
            # # spacing = nii_img.header.get_zooms()[:3]  # 只取前三个轴的 spacing
            # # print(spacing)
            # origin = nrrd_img.affine[:3, 3]
            # direction = nrrd_img.affine[:3, :3].flatten().tolist()  # 转换为列表格式
            # # print(direction)
            #
            # # 创建 SimpleITK 图像
            # sitk_img = sitk.GetImageFromArray(img_data)
            # sitk_img.SetSpacing(tuple(spacing))  # 将 spacing 转换为元组
            # sitk_img.SetOrigin(tuple(origin))  # 将 origin 转换为元组
            # sitk_img.SetDirection(direction)  # direction 已经是列表格式
            # # 定义目标图像大小、空间分辨率和方向
            # size = nii_image.GetSize()
            # spacing = nii_image.GetSpacing()
            # direction = nii_image.GetDirection()
            #
            # # 创建一个新的图像并重新采样
            # resampled_image = sitk.Resample(nii_image, size, sitk.Transform(), sitk.sitkLinear, nii_image.GetOrigin(),
            #                                 spacing, sitk.sitkIdentity)
            random_uuid = uuid.uuid4()
            output_mha_file = 'CT/' + str(random_uuid) + '.mha'
            sitk.WriteImage(sitk_img, output_mha_file)
        except:
            pass


def unzip_all_files(zip_folder, extract_to_folder):
    # 获取 ZIP 文件列表
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]

    # 确保目标解压目录存在
    if not os.path.exists(extract_to_folder):
        os.makedirs(extract_to_folder)

    # 解压每个 ZIP 文件
    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        extract_path = os.path.join(extract_to_folder, zip_file[:-4])

        # 创建解压目录
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Unzipped {zip_file} to {extract_path}")
        except Exception as e:
            print(f"Error unzipping {zip_file}: {e}")

def mhd_2_mha(folder):
    # [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk(folder) for d in dirs if d == 'segmentations']
    files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if
             file.endswith('.mhd')]
    for mhd_file in files:
        try:
            # 读取 NIfTI 文件
            sitk_img = sitk.ReadImage(mhd_file)
            # mhd_img = nib.load(mhd_file)
            #
            # # 获取图像数据、空间分辨率和方向
            # img_data = mhd_img.get_fdata()
            # spacing = [1.5, 1.5, 1.5]  # 目标间距
            # # origin = [0.0, 0.0, 0.0]  # 固定原点
            # # direction = [0.0, 0.0, 1.0,  # 固定方向矩阵
            # #                 0.0, 1.0, 0.0,
            # #                 1.0, 0.0, 0.0]
            # # spacing = nii_img.header.get_zooms()[:3]  # 只取前三个轴的 spacing
            # # print(spacing)
            # origin = mhd_img.affine[:3, 3]
            # direction = mhd_img.affine[:3, :3].flatten().tolist()  # 转换为列表格式
            # # print(direction)
            #
            # # 创建 SimpleITK 图像
            # sitk_img = sitk.GetImageFromArray(img_data)
            # sitk_img.SetSpacing(tuple(spacing))  # 将 spacing 转换为元组
            # sitk_img.SetOrigin(tuple(origin))  # 将 origin 转换为元组
            # sitk_img.SetDirection(direction)  # direction 已经是列表格式
            # # 定义目标图像大小、空间分辨率和方向
            # size = nii_image.GetSize()
            # spacing = nii_image.GetSpacing()
            # direction = nii_image.GetDirection()
            #
            # # 创建一个新的图像并重新采样
            # resampled_image = sitk.Resample(nii_image, size, sitk.Transform(), sitk.sitkLinear, nii_image.GetOrigin(),
            #                                 spacing, sitk.sitkIdentity)
            random_uuid = uuid.uuid4()
            output_mha_file = 'CT/' + str(random_uuid) + '.mha'
            sitk.WriteImage(sitk_img, output_mha_file)
        except:
            pass


def get_leaf_subdirectories(base_folder):
    leaf_folders = []

    # 遍历 base_folder 下的所有目录
    for foldername, subfolders, _ in os.walk(base_folder):
        # 如果当前目录的子目录列表为空，则说明这是最末层的目录
        if not subfolders:
            leaf_folders.append(foldername)

    return leaf_folders


def dcm_2_mha(folder):
    leaf_folders = get_leaf_subdirectories(folder)
    print(leaf_folders)
    # [shutil.rmtree(os.path.join(root, d)) for root, dirs, _ in os.walk(folder) for d in dirs if d == 'segmentations']
    # files = [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if
    #          file.endswith('.mhd')]
    for leaf_folder in leaf_folders:
        try:
            dcm_files = [os.path.join(leaf_folder, f) for f in sorted(os.listdir(leaf_folder)) if
                         f.lower().endswith('.dcm')]

            if not dcm_files:
                print("No DICOM files found.")
                continue

            # 读取 DICOM 文件系列
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dcm_files)

            # 读取 DICOM 图像
            sitk_img = reader.Execute()
            random_uuid = uuid.uuid4()
            output_mha_file = 'CT/' + str(random_uuid) + '.mha'
            sitk.WriteImage(sitk_img, output_mha_file)
        except:
            pass

def normalize_to_255(array):
    """
    将数组的值归一化到 0-255 范围。
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val) * 255
    return normalized_array.astype(np.uint8)

def make_train_data(folder):
    os.makedirs('train2', exist_ok=True)
    mha_files = [f for f in os.listdir(folder) if f.endswith('.mha')]
    for mha_file in tqdm(mha_files):
        mha_file = os.path.join(folder, mha_file)
        # print(mha_file)
        image = sitk.ReadImage(mha_file)
        # 获取图像的尺寸
        size = image.GetSize()
        # print(size)
        # 找到最小维度及其索引
        min_dim = min(size)
        # print(min_dim)
        min_dim_index = size.index(min_dim)
        # print(min_dim_index)
        # num_slices = image.GetSize()[0]
        for i in range(5, min_dim-5):
            if i % 5 == 0:
                selected_value = random.choice([-1, -2, -3, -4, -5, 1, 2, 3, 4, 5])
                if min_dim_index == 0:
                    fixed_image = sitk.GetArrayFromImage(image)[:, :, i]
                    moving_image = sitk.GetArrayFromImage(image)[:, :, i+selected_value]
                if min_dim_index == 1:
                    fixed_image = sitk.GetArrayFromImage(image)[:, i, :]
                    moving_image = sitk.GetArrayFromImage(image)[:, i+selected_value, :]
                if min_dim_index == 2:
                    fixed_image = sitk.GetArrayFromImage(image)[i, :, :]
                    moving_image = sitk.GetArrayFromImage(image)[i+selected_value, :, :]
                # fixed_image = image[:, :, i]
                #
                # moving_image = image[:, :, i+selected_value]
                # 将帧归一化到 0-255 范围
                fixed_image = normalize_to_255(fixed_image)
                # print(fixed_image.shape)
                moving_image = normalize_to_255(moving_image)
                combined_image_array = np.stack((fixed_image, moving_image), axis=-1)
                # 转换为 PIL 图像
                combined_image = Image.fromarray(combined_image_array)
                combined_image_resized = combined_image.resize((512, 512))
                # print(len(combined_image_resized.split()))
                random_uuid = uuid.uuid4()
                output_png_name = 'train2/' + str(random_uuid) + '.png'
                combined_image_resized.save(output_png_name)



if __name__ == "__main__":
    folder = 'CT'
    make_train_data(folder)