## FLARE24 Solution
##  This repository is the official implementation of [Locate, Crop and Segment: Efficient abdominal CT image segmentation on CPU](https://openreview.net/forum?id=KiRQT0mBXm&noteId=KiRQT0mBXm) of Team lyy1 on FLARE24 challenge.
## Introduction

### Overview of our work.

![](https://github.com/lay-john/FLARE24-Task2/blob/master/IMG/model.png)

![](https://github.com/lay-john/FLARE24-Task2/blob/master/IMG/liver-based-Z-axis-RoI.png)



## Environments and Requirements

The basic language for our work is [python](https://www.python.org/), and the baseline
is [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). So, you can install the nnunet frame with
the [GitHub Repository](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), or use the following comments:

```
pip install torch torchvision torchaudio
pip install -e .
```

##Dataset
You can get the dataset from [MICCAI FLARE2024](https://www.codabench.org/competitions/2320/#/pages-tab).
## Prepocessing

### convert CT images to npy

we modify the normalization function with ___preprocessing.py___,
and you could use the following comments to processing the CT images:

```
python nnunet/experiment_planning/nnUNet_convert_decathlon_task.py -i [FLARE24_Task2_imageTr_path]

python nnunet/experiment_planning/nnUNet_plan_and_preprocess -t [TASK_ID] --verify_dataset_integrity
```

It must be noted that the method is based on the __nnU-Net__, so I recommend you to convert the dataset within nnU-Net's
data preprocessing.

The usage and note concerning for ___nnUNet_convert_decathlon_task.py___ is recorded
on [website](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/dataset_conversion.md).

After preprocessing, we will obtain several folders:

```
- nnU-Net_base_folder
    - nnUNet_prepocessing
    - nnUNet_raw
    - nnUNet_trained_models
```

### Resample the data to our voxel spacing

```
python data_convert.py -nnunet_preprocessing_folder -imagesTr_floder -labelTr_floder
```
where the __nnunet_preprocessing_folder__ is the folder path of the dataset planed by nnunet. like 'nnU-Net_base_folder/nnUNet_preprocessed/Task0160_FLARE2024T2/nnUNetData_plans_v2.1_stage1'



## Training

### Train the teacher model.

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 fold --npz
```

Before Train the student model, you should move the best teacher nnunet checkpoints to replace the three files in folder [checkpoint_teacher](https://github.com/lay-john/FLARE24-Task2/tree/master/checkpoint_teacher)

### Train the student model.

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2_PPPP fold --npz
```

## Trained Models
You can get our model from the [checkpoints](https://github.com/lay-john/FLARE24-Task2/tree/master/checkpoints)
## Inference

We convert the pretrained model into onnx format for saving time. 

You can run the [nnunetv1 .model to .onnx.py](https://github.com/lay-john/FLARE24-Task2/blob/master/nnunetv1%20.model%20to%20.onnx.py) to get onnx model. The [SAVE_PATH] Should be suffixed with .onnx flavor.

```
python nnunetv1 .model to .onnx.py [SAVE_PATH]
```

Before conversion, you should move the best student nnunet checkpoints to replace the three files in folder [checkpoints](https://github.com/lay-john/FLARE24-Task2/tree/master/checkpoints).
You can to here to get [our onnx model](https://pan.baidu.com/s/1Mgu62bh-ZdaXWeNdaSkqug?pwd=8888).


Then, you can inference

```
python inference_test.py [INPUT_FOLDER] [OUTPUT_FOLDER] [SAVE_PATH]
```



## Evaluation
You can run T2andT3_FLARE24_DSC_NSD_Eval.py in [the evalution code](https://github.com/JunMa11/FLARE/tree/main/FLARE24) to get the results reported in the paper
## Results
You can get our performance in MICCAI FLARE2024 Task2 on the team named lyy1 in [result](https://www.codabench.org/competitions/2320/#/pages-tab)

## Contributing
This project is licensed under the [Apache License 2.0](https://github.com/lay-john/FLARE24-Task2/tree/master/LICENSE), and please adhere to the licenses of the models and datasets used.
## Acknowledgement

We thank the contributors of [FLARE24 datasets](https://www.codabench.org/competitions/2320/).







