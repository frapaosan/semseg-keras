# semseg-keras
A Keras implementation of PSPNet[[1]](https://arxiv.org/pdf/1612.01105.pdf) and SegNet[[2]](https://arxiv.org/pdf/1612.01105.pdf). Pascal VOC 2012[[3]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)  dataset was used. I was not able to upload the dataset here due to network constraints

## Dataset Preprocessing
When preprocessing to generate the .npy files to feed the network, refer to 'generate_x.py' and generate_y.py'. Change the variabel voc_loc to point to the location of the PASCAL VOC 2012 dataset.
```
voc_loc = "/media/francis/datasets/_AADatasets/VOC2012_orig/"
```
Also, change the resolution of the dataset to be preprocessed by changing width and height variables. For:
```
PSPNet: 473 x 473
SegNet: 256 x 256
```
You can also change how many samples you store in 1 .npy file by changing the variable num_get.
After changing the parameters, preprocess the dataset by running:
```
bash gen.sh
```

## Preprocessed Data Hierarchy
Preprocessed data should be located in the working directory as indicated below:
NOTE: FOLDER NAMES OF .NPY FILES ARE ABSOLUTE.
```
    .
    ├── Xtrain                   # Location of Xtrain_*.npy files
    │   └─── *.npy files
    ├── Ytrain                   # Location of Ytrain_*.npy files
    │   └─── *.npy files
    ├── Xval                     # Location of Xval_*.npy files
    │   └─── *.npy files
    ├── Yval                     # Location of Yval_*.npy files
    │   └─── *.npy files
    ├── gen.sh                   # Bash script for calling generate_x.py and generate_y.py
    ├── generate_x.py            # Bash script preprocessing inputs
    ├── generate_y.py            # Bash script preprocessing ground truth labels
    ├── Resnet50.py
    ├── test.py
    ├── training.py
    └── README.md
```
Sample .npy files can be downloaded [in this link.](https://drive.google.com/drive/folders/1I7fgSU4l5ptyzwd35I-5YWJSVHukmP4V?usp=sharing). The sample data are organized as follows:
```
    .
    ├── 256                   # Location of 256x256 .npy files (FOR SEGNET)
    │   └─── *.npy files      # Each .npy file has 25 samples
    └── 473                   # Location of 473x473 .npy files (FOR PSPNET)
        └─── *.npy files      # Each .npy file has 25 samples
```
After downloading, extract and put the .npy files you want to feed into the network into their respective filenames.

## Training
In line 16, '2' should be changed to how many .npy files you have in the Xtrain folder.
```
    for cnt in range(2):
```
In line 473, '50' should be changed to how total training samples you have.
```
    train_samps = 50	#1464 WHOLE SET
```
Finally, to train the network, just run
```
python training.py [model_arhitecture]
python training.py pspnet
python training.py segnet
```

The link to the pre-trained SegNet model can be downloaded [in this link.](https://drive.google.com/file/d/1aLOaiASl2KgERhOZ9iNt6D9AZgKCWDlF/view?usp=sharing)

## Evaluation and Testing
To evaluate and test the trained model, just run
```
python test.py [set_to_be_sampled] [model_architecture]
python test.py train segnet
```
For now, only the SegNet weights are available and it is somewhat biased.

### Contributions

PSPNet was my first proposal but then I also tried SegNet to check the differences. PSPNet was definitely a deeper network hence training time was recorded to be 3x longer than SegNet at 100 epochs. The PSPNet and Segnet Implementations were from [Vladkryvoruchko](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow) and [preddy5](https://github.com/preddy5/segnet) respectively. First trial of PSPNet was unsuccessful so I tried to modify the last layers to something like the last layers of SegNet since cross entropy was not working properly on 21-channel depth arrays.

## Acknowledgments
This was build for EE298-F Computer Vision Deep Learning.

## Sources
[1] Zhao, et al. "Pyramid Scene Parsing Network." Conference on Computer Vision and Pattern Recognition 2017. https://arxiv.org/pdf/1612.01105.pdf

[2] Badrinarayanan, et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." IEEE Transactions on Pattern Analysis and Machine Intelligence 2017. https://arxiv.org/pdf/1511.00561.pdf

[3] Pascal Visual Object Classes Challenge 2012. http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
