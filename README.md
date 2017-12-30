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

## Training
To train the network, just run
```
python training.py [model_arhitecture]
python training.py pspnet
python training.py segnet
```
The link to the pre-trained SegNet model can be downloaded [in this link.](http://www.dropbox.com)

## Evaluation and Testing
To evaluate and test the trained model, just run
```
python test.py [set_to_be_sampled] [model_architecture]
python test.py train segnet
python test.py val pspnet
```
The voc_loc variable here should also be changed according to where your PASCAL VOC dataset is located.

### Contributions

PSPNet was my first proposal but then I also tried SegNet to check the differences. PSPNet was definitely a deeper network hence training time was recorded to be 3x longer than SegNet at 100 epochs. The PSPNet and Segnet Implementations were from [Vladkryvoruchko](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow) and [preddy5](https://github.com/preddy5/segnet) respectively. First trial of PSPNet was unsuccessful so I tried to modify the last layers to something like the last layers of SegNet since cross entropy was not working properly on 21-channel depth arrays.

## Acknowledgments
This was build for EE298-F Computer Vision Deep Learning.

## Sources
[1] Zhao, et al. "Pyramid Scene Parsing Network." Conference on Computer Vision and Pattern Recognition 2017. https://arxiv.org/pdf/1612.01105.pdf

[2] Badrinarayanan, et al. "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." IEEE Transactions on Pattern Analysis and Machine Intelligence 2017. https://arxiv.org/pdf/1511.00561.pdf

[3] Pascal Visual Object Classes Challenge 2012. http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
