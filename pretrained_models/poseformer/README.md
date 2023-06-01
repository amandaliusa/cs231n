The demos in this directory were generated using https://github.com/zczcwh/poseformer_demo. To produce these demo outputs, clone the poseformer_demo repository and follow the setup instructions. For STS videos, comment out lines 245-246 in demo/vis_poseformer.py to avoid cropping the 2D image outputs when generating the demo videos. 

Lines to comment out:
```
edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
```