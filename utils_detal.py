import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F 


def get_angle_bw_eyes(lx, ly, rx, ry):
    w = rx - lx
    h =  ly - ry
    angle = np.arccos(w / np.sqrt(h**2 + w**2)) * 180 / np.pi
    return angle if h < 0 else -angle

def rotate_image_landmarks(image, landmarks, angle):
        transformation_matrix = torch.tensor([
            [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 
            [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
        ])

        image = torchvision.transforms.functional.rotate(image, angle, 
                                                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                        )
        new_landmarks = np.matmul(landmarks.view(-1,2), transformation_matrix)
        return image, new_landmarks.flatten()
    
def crop_resize_rotated_image(img_tensor, landmarks_coord_tensor, new_img_sizes):
    _, h, w = img_tensor.shape
    le, re, n, lm, rm = landmarks_coord_tensor.view(-1, 2)

    top_crop = min(le[1], re[1]) - 50
    left_crop = min(le[0], lm[0]) - 40
    height_crop = ((lm[1]+rm[1])/2) + 45 - top_crop
    width_crop = max(re[0], rm[0]) + 40 - left_crop
    new_img = torchvision.transforms.functional.resized_crop(img_tensor,
                                                             top=top_crop.int(),
                                                             left=left_crop.int(),
                                                             height=height_crop.int(),
                                                             width=width_crop.int(),
                                                             size=new_img_sizes
                                                             )
    return new_img

class Alignment_NN(nn.Module):
    # 10 classes for 2 coordinates of 5 landmarks
    def __init__(self,num_classes=10):
        super().__init__()
        self.model_name='resnet18'
        self.model=torchvision.models.resnet18(weights=None)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        x=self.model(x)
        return x
        
class DetectorAligner:
    def __init__(
        self, 
        yolov5_weights_path="trained_models/Yolov5_model_weights.pt", 
        alignment_model_path="trained_models/Alignment_model_statedict.pth"
        ):
        """
        Args:
            alignment_model
                PyTorch model with loaded weights
            new_img_sizes: tuple(int,int)
                Sizes the image is resized to in the end
        """
        # Detection part
        self.detection_model = torch.hub.load(r'ultralytics/yolov5', 'custom', yolov5_weights_path)
        self.detection_model.eval()

        # Alignment part
        self.alignment_model = Alignment_NN()
        self.alignment_model.load_state_dict(torch.load(alignment_model_path))
        self.alignment_model.eval()
        self.align_transforms = torchvision.transforms.Compose(
            (torchvision.transforms.ToTensor(),
             # size NN for alignment was trained on
             # hm, wm = 218, 178 
             torchvision.transforms.Resize((218, 178)))
        )

    def detect(self, image):
        """
        Function presumes there's only one face on the picture
        ----------
        Args:
            image: numpy.ndarray, path to image

        Returns: 
            cropped_img: numpy.ndarray
        """
        results = self.detection_model(image)
        # Cropping image with resulted bbox
        try:
            bboxed_image = results.crop(save=False)[0]['im'][:, :, ::-1]
        except IndexError:
            print("Model didn't detect any faces")
            return None
        return bboxed_image

    def align(self, bboxed_image, new_img_sizes=(224,224)):
        """
        Args:
            bboxed_image: numpy.ndarray
                Image after detection step
        Returns: 
            img: torch.Tensor
                Rotated then cropped and resized image based on landmarks found
        """
        # Conerting image to torch.Tensor
        # Resizing image for the model was trained on specific image size w/o augmentation
        img = self.align_transforms(bboxed_image)
        with torch.no_grad():
            landmarks = self.alignment_model(img[None, :])[0]
        # Getting rotation angle from eyes landmarks
        rotation_angle = get_angle_bw_eyes(*landmarks[0:4]).item()
        # Rotatin image and landmarks
        img, landmarks_rotated = rotate_image_landmarks(img, landmarks, rotation_angle)
        # Unscaling landmarks from interval [-1,1]
        *_, h, w = img.shape
        landmarks_rotated[0::2] = (landmarks_rotated[0::2]+0.5)*w
        landmarks_rotated[1::2] = (landmarks_rotated[1::2]+0.5)*h
        # Cropping and resizing image with given rotated landmarks
        img = crop_resize_rotated_image(
            img_tensor=img, 
            landmarks_coord_tensor=landmarks_rotated, 
            new_img_sizes=new_img_sizes
        )
        
        return img
    
    def get_face_img(self, image, new_img_sizes=(224,224)):
        """
        Function presumes there's only one face on the picture
        ----------
        Args:
            image: numpy.ndarray, path to image

        Returns: 
            face_img: torch.Tensor
        """
        face_img = self.detect(image)
        face_img = self.align(face_img, new_img_sizes)
        return face_img
    
    def get_face_img(self, image, new_img_sizes=(224,224)):
        """
        Function presumes there's only one face on the picture
        ----------
        Args:
            image: numpy.ndarray or path to an image

        Returns: 
            face_img: torch.Tensor or None
        """
        face_img = self.detect(image)
        if face_img is not None:
            face_img = self.align(face_img, new_img_sizes)
        return face_img
            