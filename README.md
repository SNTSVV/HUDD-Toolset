

# HUDD (Heatmap-based Unsupervised Debugging of DNNs)

This repository contains the tools and the data appearing in: 
* "**Supporting DNN Safety Analysis and Retraining Through Heatmap-Based Unsupervised Learning**" by Hazem Fahmy, Fabrizio Pastore, Mojtaba Bagherzadeh, and Lionel Briand, published in the [IEEE Transactions on Reliability](https://ieeexplore.ieee.org/abstract/document/9439863); available for [download](https://arxiv.org/abs/2002.00863).
* "**HUDD: A tool to debug DNNs for safety analysis**" by Hazem Fahmy, Fabrizio Pastore, and Lionel Briand, presented in [IEEE/ACM 44th International Conference on Software Engineering](https://ieeexplore.ieee.org/document/9793750); available for [download](https://arxiv.org/abs/2210.08356).

# Project Description
We observe three major challenges with existing practices regarding DNNs in safety-critical systems:
* Scenarios that are under-represented in the test set may lead to serious safety violation risks but may, however, remain unnoticed; 
* characterizing such high-risk scenarios is critical for safety analysis; 
* retraining DNNs to address these risks is poorly supported when causes of violations are difficult to determine. 

To address these problems in the context of DNNs analyzing images, we propose Heatmap-based Unsupervised Debugging of DNNs (HUDD), an approach that automatically supports the identification of root causes for DNN errors. HUDD identifies root causes by applying a clustering algorithm to heatmaps capturing the relevance of every DNN neuron on the DNN outcome. Also, HUDD retrains DNNs with images that are automatically selected based on their relatedness to the identified image clusters. We evaluated HUDD with DNNs from the automotive domain. HUDD was able to identify all the distinct root causes of DNN errors, thus supporting safety analysis. Also, our retraining approach has shown to be more effective at improving DNN accuracy than existing approaches.

# Usage

## DataSets and DNNs

All case studies and DNNs can be downloaded from [here](https://zenodo.org/record/5725116#.Yyc7x-xBzuU)
The package contains the following files:

  - OC.tar -> [OC](https://paperswithcode.com/dataset/oc) case study -> Detection of a driver's open/closed eyes (2 classes)
  - GD.tar -> [GD](https://paperswithcode.com/dataset/gd) case study -> Detection of a driver's eyes' gaze direction (8 classes)
  - HPD.tar -> [HPD](https://paperswithcode.com/dataset/hpd) case study -> Detection of a driver's head pose direction (9 classes)
  - OD.tar -> [OD](https://paperswithcode.com/dataset/celeba) case study -> Detection of absense/presence of an eye glasses (2 classes)
  - TS.tar -> [TS](https://paperswithcode.com/dataset/gtsrb) case study -> Detection of road traffic signs
  - FLD.tar -> FLD case study -> Detection of 27 facial land marks of a driver's face (nose, eyes, mouth)
  - FLD_IEEPackage.tar.bz2 -> Additional data required to perform landmark detection for FLD case study.
  - RQ2.tar.bz2 -> Toolset to collect data for RQ2 (Python binary files)

Each of these tar files contains compressed folders, as follows:

- xx_clusterImages.tar.bz2 -> Root cause clusters generate by HUDD
- xx_BL1.tar.bz2 -> Models retrained using BL1
- xx_BL2.tar.bz2 -> Models retrained using BL2
- xx_HUDD.tar.bz2 -> Models retrained using HUDD
- xx_OriginalModel.pth -> Original model before retraining
- xx_TestSet.tar.bz2 -> Test set images + labels
- xx_TrainingSet.tar.bz2 -> Training set images + labels
- xx_ImprovementSet.tar.bz2 -> ImprovementSet images + labels

## HUDD Architecture and Code Contents

![img1](./images/HUDD_tool.png)

* The HUDD tool consists of a command line user interface called (./HUDD.py) and five modules: (./testModule.py), (./heatmapModule.py), (./clusterModule.py), (./assignModule.py), (./retrainModule.py).

* To execute HUDD, the engineer provides to the (./Helper.py) the DNN model to be analyzed. The DNN under analysis shall be stored in the DNNModels folder; the datasets shall be provided in the DataSets folders TrainingSet, TestSet, and ImprovementSet.

* The [DNN Testing Module](testModule.py) uses the DNN under analysis to process the inputs in the training and test set. Outputs are exported in the files trainResult.csv and testResult.csv. The latter is used to determine which are the error-inducing images to be used to generate RCCs (Step 1).

* The [Heatmaps Module](heatmapModule.py) generates heatmaps for error-inducing images. For each DNN layer, it stores, in the Heatmaps directory, a NumPy file with the heatmaps of all the error-inducing images.

* The [Clustering Module](clusterModule.py), for each layer, computes the distance matrix and exports it in an XLSX file. Also, it performs hierarchical agglomerative clustering based on the heatmaps generated for each layer and selects the optimal number of clusters. Finally, for each ??????????? layer, it stores the generated clusters in a directory called T/ClusterAnalysis/LayerK. The clusters for the layer showing the best results (layer ???? ) are copied in the parent folder (i.e., ./T/LayerX/ ). For each RCC, the ClusterModule generates a directory with all the images belonging to the cluster, which are to be visually inspected by engineers as per HUDD Step 2.

* The [Assignment Module](assignModule.py) processes the ImprovementSet images and stores the unsafe set in the folder UnsafeSet. Finally, the [Retraining Module](retrainModule.py) retrains the DNN using the images in the training and unsafe sets. The retrained DNN model is saved in the DNNModels directory.

## Usage

![img2](./images/RealHPDClusters.png)

For each case study, to replicate the whole experient (i.e., generation of root cause clusters + retraining) please proceed as follows.

1. Create a directory (e.g. ./TR_Package/)
2. Create a case study folder (e.g. ./TR_Package/OC/)
3. Create a folder DataSets (e.g. ./TR_Package/OC/DataSets/)
4. Create a folder DNNModels (e.g. ./TR_Package/OC/DNNModels/)
5. Unzip TrainingSet, TestSet, ImprovementSet in DataSets (e.g. ./TR_Package/OC/DataSets/TrainingSet)
6. Put the original DNN model in DNNModels (e.g. ./TR_Package/OC/DNNModels/OriginalModel.pth)
7.

```
	python HUDD.py -o directoryPath -m modelName -mode retrainMode
```

directoryPath is the path to the folder containing the case study data (e.g. ./TR_Package/OC)
modelName is the model to be used in the DNNModels folder (e.g. OriginalModel.pth)
retrainMode is the mode to retrain the model (e.g. HUDD - BL1 - BL2)

Below, we report the case study names that we used in our setup:
./TR_Package/OC
./TR_Package/GD
./TR_Package/OD
./TR_Package/TS
./TR_Package/HPD
./TR_Package/FLD

Note:
For FLD case study:
	- Put IEEPackage folder in ./TR_Package/FLD/
	- Put ieetrain.npy, ieetest.npy, ieeimprove.npy in ./TR_Package/FLD/IEEPackage/



## Regeneration of RQ1 raw results (without replicating the whole study)


1. Create a directory (e.g. ./TR_Package/)
2. Create a case study folder (e.g. ./TR_Package/OC/)
3. Create a folder DataSets (e.g. ./TR_Package/OC/DataSets/)
4. Create a folder DNNModels (e.g. ./TR_Package/OC/DNNModels/)
5. Unzip TrainingSet, TestSet, ImprovementSet in DataSets (e.g. ./TR_Package/OC/DataSets/TrainingSet)
6. Unzip HUDD, BL1, BL2 models in DNNModels (e.g. ./TR_Package/OC/DNNModels/BL1_1_92.37.pth)
7.
```
	python HUDD.py -o directoryPath -m modelName -rq True
```

## Regeneration of RQ2 results (without replicating the whole study)

1. Create a directory (e.g. ./TR_Package/)
2. Create a case study folder (e.g. ./TR_Package/OC/)
3. Create a folder DataSets (e.g. ./TR_Package/OC/DataSets/)
4. Create a folder DNNModels (e.g. ./TR_Package/OC/DNNModels/)
5. Unzip TrainingSet, TestSet, ImprovementSet in DataSets (e.g. ./TR_Package/OC/DataSets/TrainingSet)
6. Unzip HUDD, BL1, BL2 models in DNNModels (e.g. ./TR_Package/OC/DNNModels/BL1_1_92.37.pth)
7. Unzip RQ2.zip
8.
```
	python RQ2.py -o directoryPath -m modelName
```
directoryPath is the path to the folder containing case studies (e.g. ./TR_Package/OC)
modelName is the model to be tested in the DNNModels folder (e.g. BL1_1_92.37.pth)

Note:
For FLD case study, please put ieetest.npy folder in ./TR_Package/FLD/DataSets/

# Reference:

If you use our work, please cite HUDD in your publications. Here is an example BibTeX entry:
```
@ARTICLE{Fahmy_HUDD,  
author={Fahmy, Hazem and Pastore, Fabrizio and Bagherzadeh, Mojtaba and Briand, Lionel},  
journal={IEEE Transactions on Reliability},   
title={Supporting Deep Neural Network Safety Analysis and Retraining Through Heatmap-Based Unsupervised Learning},   
year={2021},  
volume={70},  
number={4},  
pages={1641-1657},  
doi={10.1109/TR.2021.3074750}
}
```

OR

```
@inproceedings{Fahmy_HUDD_tool,
author = {Fahmy, Hazem and Pastore, Fabrizio and Briand, Lionel},
title = {HUDD: A Tool to Debug DNNs for Safety Analysis},
year = {2022},
isbn = {9781450392235},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3510454.3516858},
doi = {10.1145/3510454.3516858},
booktitle = {Proceedings of the ACM/IEEE 44th International Conference on Software Engineering: Companion Proceedings},
pages = {100???104},
numpages = {5},
series = {ICSE '22}
}
```

# Acknowledgment:

This project has received funding from IEE Luxembourg, Luxembourg???s National Research Fund (FNR) under grant BRIDGES2020/IS/14711346/FUNTASY, the European Research Council (ERC) under the European Union???s Horizon 2020 research and innovation programme (grant agreement No 694277), and NSERC of Canada under the Discovery and CRC programs. Authors would like to thank Thomas Stifter and Jun Wang from IEE for their valuable support.
The experiments presented in this project were carried out using the HPC facilities of the University of Luxembourg (see http://hpc.uni.lu).
