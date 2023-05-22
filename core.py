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

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

HOME = os.getcwd()
print(HOME)


#from IPython import display
# display.clear_output()

ultralytics.checks()

# print("Downloading test datasets.")

# if not os.path.exists(f"{HOME}/datasets"):
#     os.makedirs(f"{HOME}/datasets")

# rf = Roboflow(api_key="Hp5hidJ2MxhXMd3IYOmb")
# project = rf.workspace("samdeploymentmodel").project("pcb_quality_control")
# dataset = project.version(1).download("yolov8")

# SAM_weights_path = "SAM_weights"
sam_checkpoint_path = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
model_checkpoints_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

print("Looking for SAM_weights.")
# if not os.path.exists(SAM_weights_path):
#     os.mkdir(SAM_weights_path)

if not os.path.exists("sam_vit_h_4b8939.pth"):
    print("SAM_weights not found! Downloading...")

    # if not os.path.exists(SAM_weights_path):
    #     os.mkdir(SAM_weights_path)

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

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    return mask_image


def polygon_area(polygon):
    num_points = len(polygon)
    signed_area = 0

    for i in range(num_points):
        x_i, y_i = polygon[i]
        x_next, y_next = polygon[(i + 1) % num_points]
        signed_area += x_i * y_next - x_next * y_i

    return abs(signed_area) / 2


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
                                    # save=True
                                    )

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

    for mask, cls in zip(sorted_masks, sorted_class_p):
        sam_img = show_mask(mask.cpu().numpy(), plt.gca(), cls=cls)

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

    return df, sam_img  # , yolo_image  # , sam_image,
