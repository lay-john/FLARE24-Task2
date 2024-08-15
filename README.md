#  Solution of Team lyy1 for FLARE24 Task 2 Challenge

## Introduction

### Overview of our work.

![](E:\graduate\medicalImage\nnUNet\nnUNet\IMG\model.png)

![](E:\graduate\medicalImage\nnUNet\nnUNet\IMG\liver-based Z-axis RoI.png)



## Environments and Requirements

The basic language for our work is [python](https://www.python.org/), and the baseline
is [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). So, you can install the nnunet frame with
the [GitHub Repository](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1), or use the following comments:

```
pip install torch torchvision torchaudio
pip install -e .
```

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

Before Train the student model, you should move the best teacher nnunet checkpoints to replace the three files in folder __'checkpoint_teacher'__.

### Train the student model.

```
python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2_PPPP fold --npz
```



## Inference

We convert the pretrained model into onnx format for saving time. 

You can run the nnunetv1 .model to .onnx.py to get onnx model. The [SAVE_PATH] Should be suffixed with .onnx flavor.

```
python nnunetv1 .model to .onnx.py [SAVE_PATH]
```

Before the Inference, you should move the best student nnunet checkpoints to replace the three files in folder __'checkpoints'__.



Then, you can inference

```
python inference_test.py [INPUT_FOLDER] [OUTPUT_FOLDER] [SAVE_PATH]
```



You also can run the inference.py to inference.

```
python inference.py [INPUT_FOLDER] [OUTPUT_FOLDER]
```
Before the Inference, you should move the best student nnunet checkpoints to replace the three files in folder __'checkpoints'__.



## Evaluation

## Results


## Acknowledgement

MACCAI FLARE2024 Task 2 https://www.codabench.org/competitions/2320/


## What's New?




