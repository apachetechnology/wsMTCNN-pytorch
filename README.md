# mtcnn-pytorch
pytorch implementation of  face detection algorithm  MTCNN

### Usage MTCNN

Just download the repository and then do this

```
from model.FaceDetector import CFaceDetector
from model.utils import show_bboxes
from PIL import Image

image = Image.open('images/test3.jpg')

obj = CFaceDetector()
bounding_boxes, landmarks = obj.detect_faces(image)
image = show_bboxes(image, bounding_boxes, landmarks)
image.show()
```

### Requirements

- pytorch 0.4
- Pillow, numpy

### Credit

- [pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
- https://github.com/kpzhang93/MTCNN_face_detection_alignment
- https://github.com/TropComplique/mtcnn-pytorch
- https://github.com/polarisZhao/mtcnn-pytorch

### Reference

**MTCNN:** [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

