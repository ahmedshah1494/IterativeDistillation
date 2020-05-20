# IterativeDistillation

## Organization :
The main code files are :
* `distillation.py` which contains the compression algorithms
* `models.py` for the model architectures
* `training.py` to train uncomprssed teacher models
Other files are mostly test scripts used through the paper.


## Usage
To use the code, run the `distillation.py` file with all needed options (specified in file).
Example : 

`python training.py --dataset cifar10 --datafolder path/to/data --model vgg16 --outfile path/to/model.pt`
`python distillation.py --dataset cifar10 --datafolder path/to/data --teacher_model_file path/to/model.pt`

## Requirements
* python>=3.7
* pytorch>=1.3.1
* cuda>=9.2
* torchvision>=0.2.2 
* tqdm>=4.41.1
* pillow==6.1.0
