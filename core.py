import pandas as pd
# from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO
import os
import requests
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

HOME = os.getcwd()
print(HOME)


#from IPython import display
# display.clear_output()

ultralytics.checks()

# SAM_weights_path = "SAM_weights"
sam_checkpoint_path = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
model_checkpoints_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

print("Looking for SAM_weights.")

if not os.path.exists("sam_vit_h_4b8939.pth"):
    print("SAM_weights not found! Downloading...")

    try:
        response = requests.get(model_checkpoints_url)
        response.raise_for_status()

        with open(sam_checkpoint_path, "wb") as file:
            file.write(response.content)

        print(f"SAM {model_type} model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")

print("Loading SAM model.")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
sam.to(device="cuda") if torch.cuda.is_available() else sam.to()

print("Creating SAM embeddings.")
predictor = SamPredictor(sam)


def show_mask(mask, ax, cls=1, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if cls == 1:  # color for voids
            color = np.array([30/255, 144/255, 255/255, 0.7])
        else:
            color = np.array([255/255, 0.0, 0.0, 0.6])
            
            
   # color_rgb = color[:3]  # Extract the first three channels (red, green, blue)
    #alpha = color[3]  # Extract the alpha channel
    
    # Multiply the RGB values by the alpha channel
    #color_rgb *= alpha
    
    # Convert the RGB values to the range [0, 255]
    #color_rgb *= 255
   # color_rgb = color_rgb.astype(np.uint8)

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #return ax.imshow(mask_image)
    
    return mask_image
    


def polygon_area(polygon):
    num_points = len(polygon)
    signed_area = 0

    for i in range(num_points):
        x_i, y_i = polygon[i]
        x_next, y_next = polygon[(i + 1) % num_points]
        signed_area += x_i * y_next - x_next * y_i

    return abs(signed_area) / 2
    
def remove_dir(directory):

    # Check if the directory exists
    if os.path.exists(directory):
        # Remove all files and subdirectories within the directory
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        
        # Remove the empty directory
        os.rmdir(directory)


print("Loading Yolo model.")
model_name_path = "yolov8_pcb_best.pt"
model = YOLO(model_name_path)


def process(uploaded_image, uploaded_file_name):
    # image_path = './img_test/22.jpg'

    # image = cv2.imread(uploaded_image)

    image = np.array(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    print("Predicting with Yolo model.")
    results_for_sam = model.predict(image,  # source=image_path,
                                    conf=0.25,
                                    save=True
                                    )
    
    img_dir = HOME + "/runs/detect/predict/"          
    image_Yolo = Image.open(img_dir + "image0.jpg")
    #image_Yolo = cv2.cvtColor(image_Yolo, cv2.COLOR_BGR2RGB)
    remove_dir(img_dir)

    input_box = results_for_sam[0].boxes.xyxy.cpu().data.numpy()
    class_p = results_for_sam[0].boxes.cls.cpu().data.numpy()

    input_boxes = torch.tensor(input_box, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    masks.shape

    sorted_indices = np.argsort(class_p)

    sorted_masks, sorted_class_p = masks[sorted_indices], class_p[sorted_indices]

    sam_img = image.copy()
    #sam_img_ov = np.full([1004,1004,1], 0.0)
    #sam_img = np.dstack((sam_img, sam_img_ov))
    sam_img_list = list()
    #print(sam_img.shape)
    for mask, cls in zip(sorted_masks, sorted_class_p):
        sam_img2 = show_mask(mask.cpu().numpy(), plt.gca(), cls=cls)
        #print(sam_img2.shape)
        #sam_img += sam_img2.astype(np.uint8)
        sam_img_list.append(sam_img2)
        
        
    #sam_img = np.vstack(sam_img_list)
    sam_img = np.zeros_like(sam_img_list[0], dtype=np.float32)
    # Iterate over the image list and overlay each image onto the overlayed_image
    for image in sam_img_list:
        #sam_img = cv2.addWeighted(sam_img, 1.0, image, 1.0, 0.0, dtype=cv2.CV_32F)
        sam_img = cv2.add(sam_img, image, dtype=cv2.CV_32F)
    
    # Convert the overlayed_image to the appropriate data type (e.g., uint8) for visualization
    #sam_img = np.clip(sam_img, 0, 255).astype(np.uint8)
    
    # Normalize the overlayed_image to [0, 255]
    sam_img = cv2.normalize(sam_img, None, 0, 255, cv2.NORM_MINMAX)

# Convert the overlayed_image to the appropriate data type (e.g., uint8) for visualization
    sam_img = sam_img.astype(np.uint8)

    
    print(sam_img.shape)
        

    component_total_area = 0
    void_total_area = 0
    biggest_void_area = 0

    for mask, cls in zip(masks, class_p):
        polygon = np.argwhere(mask.cpu().data.numpy())

        # Calculate the area
        area = polygon_area(polygon[:, 1:])

        if cls == 0:  # component
            component_total_area += area
        elif cls == 1:  # void
            void_total_area += area
            if area > biggest_void_area:
                biggest_void_area = area

    print(f'Component_total_area: {component_total_area}')
    print(f'void_total_area: {void_total_area}')

    print(f'biggest_void_area: {biggest_void_area}')
    print(f'Total void%: {void_total_area / component_total_area}')

    data = {
        'Image': [uploaded_file_name],
        'Component_total_area': [component_total_area],
        'void_total_area': [void_total_area],
        'Max.void': [biggest_void_area],
        'Void%': [void_total_area / component_total_area]
    }

    df = pd.DataFrame(data)
    print(df.head())

    return df, image_Yolo, sam_img  # , yolo_image  # , sam_image,
    
    
def get_system_info():
    import psutil

    # Get CPU information
    cpu_count = psutil.cpu_count(logical=False)  # Number of physical CPUs
    cpu_threads = psutil.cpu_count(logical=True)  # Number of logical CPUs
    
    # Get RAM information
    ram_total = psutil.virtual_memory().total  # Total RAM in bytes
    
    # Get disk information
    disk_usage = psutil.disk_usage('/')  # Disk usage of the root directory
    disk_total = disk_usage.total  # Total disk space in bytes
    disk_used = disk_usage.used  # Used disk space in bytes
    
    # Print the information
    #print(f"CPU Count: {cpu_count}")
    #print(f"CPU Threads: {cpu_threads}")
    #print(f"Total RAM: {ram_total / (1024**3):.2f} GB")
    #print(f"Total Disk Space: {disk_total / (1024**3):.2f} GB")
    #print(f"Used Disk Space: {disk_used / (1024**3):.2f} GB")
    
    return f'System info: {cpu_threads} Threads, {cpu_count} CPUs, {ram_total / (1024**3):.2f} GB, {disk_used / (1024**3):.2f}/{disk_total / (1024**3):.2f} GB'

