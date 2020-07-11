# Robust Template Matching Using Scale-Adaptive Deep Convolutional Features
Pytorch Unofficial Implementation of Robust Template Matching Using Scale-Adaptive
Deep Convolutional Features

http://www.apsipa.org/proceedings/2017/CONTENTS/papers2017/14DecThursday/TA-02/TA-02.5.pdf

# Requirements
* torch (1.0.0)
* torchvision (0.2.1)
* cv2
* (optional) cython

# Usage
```
python run.py [sample_image_path] [template_image_path] --use_cuda --use_cython
```

* add --use_cuda option to use GPU
* add --use_cython option to execute with cython

result image will be saved as result.png

# Example
```
python run.py sample/sample1.jpg template/template1.png --use_cuda --use_cython
```

|sample image|template image|result image|
|---|---|---|
|<img src="https://i.imgur.com/yYhdis1.png" width=300>|<img src="https://i.imgur.com/XT8Powb.png" width=70>|<img src="https://i.imgur.com/PbAJ7yq.png" width=300>|

```
python run.py sample/sample2.jpg template/template2.png --use_cuda --use_cython
```

|sample image|template image|result image|
|---|---|---|
|<img src="https://i.imgur.com/KEDIu1p.jpg" width=300>|<img src="https://i.imgur.com/nXRvBjU.png" width=70>|<img src="https://i.imgur.com/nqMdhbX.jpg" width=300>|
