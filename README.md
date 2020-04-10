# Pytorch-Uodac
## Installation

### Requirements

- Linux OS: Ubuntu 16.04
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

### Install pytorch-uodac

Clone the Pytorch-Uodac repository.
```shell
git clone https://github.com/ChenYingpeng/Pytorch-Uodac
cd Pytorch-Uodac
```

Install build requirements and then install pytorch-uodac.
  Note:(We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

```shell
pip install -r requirements.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
python3 setup.py develop
```

### Generate train and test json data
Generate tran json data
```shell
python3 tools/data_process/generate_train_json.py --xml-dir [xxx]  --json [xxx]
```
Example
```shell
python3 tools/data_process/generate_train_json.py --xml-dir ../underwater/optics/data/train/box/  --json ../underwater/optics/data/train/train_data_annotations.json
```
Generate test json data

```shell
python3 tools/data_process/generate_test_json.py --test-image-dir [xxx]  --save-json-path [xxx]
```
Example
```shell
python3 tools/data_process/generate_test_json.py --test-image-dir ../underwater/optics/data/test-A-image/  --save-json-path ../underwater/optics/data/annotations/test-A-image.json
```

### Train
```shell
./tools/dist_train.sh configs/underwater/optics/xxxx.py 2
```
### Submit
```shell
./tools/dist_submit.sh configs/underwater/optics/xxxx.py ../underwater/optics/output/xxxx/latest.pth 2 --format_only
```
You could find `test_A_image_submission.csv` on `submit/`.
