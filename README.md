# TDT4265 AdaIN
PyTorch implementation of Adaptive instance normalization for the course "Computer Vision and Deep Learning" (TDT4265) at NTNU. Thanks to naoto0804 for the project for the implementation of [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git). Documentation is added where we think it's necessary.

## Requirements
- Python 3.5+
- PyTorch 0.4+
- TorchVision
- Pillow

(for training)
- tqdm
- TensorboardX

(for visualization)
- Numpy
- Matplotlib
- Tensorflow (for the tensorboard backend)
## Usage
(Also refer to the documentation of [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN.git))
### Models
All weight files (.pth) which are relevant for the project at NTNU are part of the repository and can be found in `./weights/` and `./experiments/<experiment name>/`. It should not be necessary to download additional models or call the file `torch_to_pytorch.py`.
### Testing
For a list of all options, type:
```sh 
$ python3 test.py --help
```
Here are some exemplary commands, that we used for testing. For testing a single file, you may type:
```sh
$ python3 test.py --content <image file> --style <image file> \
  --dec <decoder architecture> --dev_w <weights file>
```
Testing on multiple files (e.g. for speed testing or average loss metrics), without saving the files.
```sh
$ python3 test.py --content_dir <content directory> --style_dir <style directory>  \
  --dec <decoder architecture> --dec_w <weights file> --only_loss --crop 
```
Note that the `--dec` option only takes one of four architectures,
- `resnet18`,
- `inceptionv3`,
- `VGG19` and
- `VGG19B` (the uncropped VGG19).
When `--dec` is given, `--dec_w` should point to a .pth weights file that fits this architecture. The default is `VGG19` and `./weights/vgg_normalised.pth`, which is the pretrained model from the original implementation.
### Training
For a list of all options, type:
```sh 
$ python3 train.py --help
```
Training can be started by typing:
```sh
$ python3 train.py --content_dir <content directory> --style_dir <style directory> --dec <decoder architecture> --max_iter <max mumber of iterations> --name <experiment name>
```
Any output of this command is written to  `./experiments/<experiment name>`.
### Visualization
Progress logs can be plotted for each experiment. For a list of all options, type:
```sh 
$ python3 tensorflow_log_loader.py --help
```
To do this, type:
```sh
$ python3 tensorflow_log_loader.py --log_name experiments/<experiment name>/events.out.tfevents.<ID> \ 
  --save --save_dir experiments/mini_datasets/ --linear
```
