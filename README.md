# Synthetic Plants

The world is advancing towards automation of every task, including agri- culture, such as selective weeding, and this will have a huge impact on the farm productivity on profitability. The most powerful tools to let autonomous robots to properly perceive the environment are neural networks. Unfortunately, such methods are data driven, meaning that they rely on a large labeled datasets, and the collection and preprocess of this data is expansive, time consuming, and depend heavily on manual labeling.

This project proposes a pipeline for image dataset augmentation generating synthetic data using generative adversarial networks. It starts generating the mask of the crop from random noise, and from the mask generates the final RGB and NIR images. For the mask generation it uses DCGAN network and for the final images uses SPADE network. The images are generated in a two step process in order to achieve the RGB, NIR, and semantic segmentation of each plant. One possibility would be generating the RGB and NIR images directly from random noise, but depending on the project that this dataset will be used, the semantic segmentation may also be needed.

The process is validated using metrics that compare the quality of the gen- erated images with the original dataset and also a semantic segmentation model that evaluates the difference between using the original dataset compared with the synthetic and augmented dataset. The metrics show that the values of the synthetic images converge to the values of the original ones as the quality improves over the training and also that the segmentation performance can indeed be improved by using the new augmented dataset.

<p align="center">
<img src="report/images/pca.png" width="900"/><br>
<img src="report/images/mpt.png" width="900"/><br>
<img src="report/images/history.png" width="900"/><br>
<img src="report/images/dca.png" width="900"/><br>
</p>

## Mask Image Generation
<p align="center">
[![Mask Image Generation](https://i.ytimg.com/vi/v2xjxWj6xKI/1.jpg)](https://www.youtube.com/watch?v=v2xjxWj6xKI)
</p>

## RGB Image Generation
[![RGB Image Generation](https://i.ytimg.com/vi/6gSF-rcAYKI/1.jpg)](https://www.youtube.com/watch?v=6gSF-rcAYKI)

## NIR Image Generation
[![NIR Image Generation](https://i.ytimg.com/vi/v6mq-mdmbDI/1.jpg)](https://www.youtube.com/watch?v=v6mq-mdmbDI)

## Instructions

* Generate dataset for GAN

`python preprocess.py --dataset_path --annotation_path --plant_type --dimension --background --blur`

* Train DCGAN network (GAN for mask images)

```
cd stage_1
python main.py --dataset_path
```

* Train SPADE network (GAN for RGB and NIR images)

```
cd stage_2
python main.py --dataset_path
```

* Generate dataser for Segmentation

```
cd stage_2
python create_dataset.py --dataset-path  --annotation_path --output_path --background --blur
```

* Train Segmentation

```
cd segmentation
python segmentation.py --dataset-path  --dataset_type
```
