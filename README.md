# MediaPipe Hand Crop Fix

Code for "Optimizing Hand Area Detection in MediaPipe Holistic Full-Body Pose Estimation to Improve Accuracy and Prevent Downstream Errors".

Fixing https://github.com/google/mediapipe/issues/5373

## Usage

```bash
pip install git+https://github.com/sign-language-processing/mediapipe-hand-crop-fix
```

Download the Panoptic Hand Pose Dataset:
```bash
cd data
wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand143_panopticdb.tar
wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip
```

