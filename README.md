## PCBQualityControl

Quality control is of vital importance during electronics production. As the methods of producing electronic circuits improve, there is an increasing chance of solder defects during assembling the printed circuit board (PCB). Technology like X-ray imaging is used for inspection.

AI-based models are proposed in the state-of-the art.

We use one of the latest segmentation models to solve this problem of void detection.

SAM (Segment Anything Model) It is agnostic model that can segment every single region in the image as a new class, using a point or surrounding the target zones.

## Technologies

| Technology   | Description                                                               |
|--------------|---------------------------------------------------------------------------|
| Python       | An interpreted, high-level programming language used for general-purpose programming. |
| Segment Anything Model (SAM) | A new AI model from Meta AI that can "cut out" any object, in any image, with a single click. |
| Yolov8       | An anchor-free, real-time object detection model that can achieve state-of-the-art accuracy and speed. |
| OpenCV | A free and open-source library of programming functions mainly for real-time computer vision. |
| Roboflow | An end-to-end computer vision platform that makes it easy to build, train, and deploy computer vision models. |

## Project steps
Part 1 - Completed on Roboflow

  ■ Annotation
  
  ■ Augmentation

Part 2 - 01_Yolo_Training_PCBQualityControl.ipynb
  ■ Yolo training on two classes:

  voids
  
  component (darker background)
  
  ■ Yolo validation

Part 3 - 02_Inference_SAM_PCBQualityControl.ipynb
  ■ Using a pre-trained SAM to segment voids and background, using the output of yolo:

  Input: image and corresponding bounding boxes given by yolo as output

  Output: segmented areas with two different masks

![image](https://github.com/ajosegun/PCBQualityControl/assets/94995067/48bbedd5-637c-4654-8cd7-578d7dac0389)

## Getting Started
To get started with this project, you will need to have the following installed:

Python 3.6 or higher


## Installation
Once you have installed the necessary dependencies, you can clone the project repository from GitHub:

```
git clone https://github.com/ajosegun/PCBQualityControl.git
```

The code is divided into 2 notebooks.
1. 01_Yolo_Training_PCBQualityControl.ipynb
  
2. 02_Inference_SAM_PCBQualityControl.ipynb


## Conclusion
The Pix2Pix GAN is a powerful tool for image-to-image translation. With this model, you can translate images from one domain to another, such as from daytime to nighttime, or from sketches to photographs. The model is easy to use and can be trained on a variety of different datasets.







