import math
import numpy as np

import cv2
from PIL import Image

import torch
from torch.nn.functional import interpolate
from .networks import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess

class CFaceDetector:
    def __init__(self):
        print('CFaceDetector is initialized')

    def detect_faces(self, image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8],
                    nms_thresholds=[0.7, 0.7, 0.7]):
        pnet, rnet, onet = PNet(), RNet(), ONet()
        onet.eval()

        width, height = image.size
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        scales = []
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        bounding_boxes = []
        for s in scales:    # run P-Net on different scales
            boxes = self.run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(
            bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        output = rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = torch.FloatTensor(img_boxes)
        output = onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(
            xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(
            ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    # Run P-Net, generate bounding boxes, and do NMS.
    def run_first_stage(self, image, net, scale, threshold):
        width, height = image.size
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')
        img = torch.FloatTensor(_preprocess(img))

        output = net(img)
        probs = output[1].data.numpy()[0, 1, :, :]
        offsets = output[0].data.numpy()

        boxes = self._generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    # Generate bounding boxes at places where there is probably a face.
    def _generate_bboxes(self, probs, offsets, scale, threshold):
        stride = 2
        cell_size = 12

        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images, so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride*inds[1] + 1.0)/scale),
            np.round((stride*inds[0] + 1.0)/scale),
            np.round((stride*inds[1] + 1.0 + cell_size)/scale),
            np.round((stride*inds[0] + 1.0 + cell_size)/scale),
            score, offsets
        ])

        return bounding_boxes.T
    
    def imresample(self, img, sz):
        im_data = interpolate(img, size=sz, mode="area")
        return im_data

    def crop_resize(self, img, box, image_size):
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = cv2.resize(
                img,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        elif isinstance(img, torch.Tensor):
            img = img[box[1]:box[3], box[0]:box[2]]
            out = self.imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(),
                (image_size, image_size)
            ).byte().squeeze(0).permute(1, 2, 0)
        else:
            out = img.crop(box).copy().resize(
                (image_size, image_size), Image.BILINEAR)
        return out
    
    """Extract face + margin from PIL Image given bounding box.
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    def extract_face(self, oImg, aBox, image_size=400, margin=0, save_path=None):
        margin = [
            margin * (aBox[2] - aBox[0]) / (image_size - margin),
            margin * (aBox[3] - aBox[1]) / (image_size - margin),
        ]

        raw_image_size = oImg.size
        newBox = [
            int(max(aBox[0] - margin[0] / 2, 0)),
            int(max(aBox[1] - margin[1] / 2, 0)),
            int(min(aBox[2] + margin[0] / 2, raw_image_size[0])),
            int(min(aBox[3] + margin[1] / 2, raw_image_size[1])),
        ]

        oFace = self.crop_resize(oImg, newBox, image_size)
        
        # save_img(face, save_path)

        # face = F.to_tensor(np.float32(face))

        return oFace

