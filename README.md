# Vineyard Disease Classification

Deep learning-based grapevine leaf disease classification and embedded deployment, developed as part of the Master's dissertation of **Patrícia Sergueevna Moltchanova** at the University of Aveiro.

## Overview

This repository contains the code and supporting material developed for camera-based grapevine leaf disease classification using deep learning models.

This work has two different approachs:
- **Selective binary classification of Black Rot**
- **Multiclass grapevine disease classification and embedded deployment**

The first stage focused on detecting Black Rot against a heterogeneous group of non-Black Rot samples. The second stage extended the problem to the classification of five grapevine leaf conditions and included robustness analysis, model conversion, quantisation and deployment on an embedded camera-based platform.

-- 

## 1. Binary Black Rot Classification

The first stage of the work addressed the selective detection of Black Rot.

The classification problem was defined as:

- **Positive class:** Black Rot
- **Negative class:** Healthy Leaf, Esca, Downy Mildew and Powdery Mildew

A total of four convolutional neural network architectures were evaluated using transfer learning:

- MobileNetV2
- DenseNet121
- VGG16
- ResNet50

Among the evaluated architectures, **ResNet50** achieved the highest overall performance, with an accuracy of **0.989**.

This stage demonstrated the feasibility of selectively detecting Black Rot. However, because all other conditions were grouped into a single non-Black Rot class, the approach was subsequently extended to multiclass classification.

---

## 2. Multiclass Grapevine Disease Classification

The multiclass stage considers five classes:

- Healthy Leaf
- Black Rot
- Esca
- Downy Mildew
- Powdery Mildew

The following lightweight architectures were evaluated:

- MobileViT-XXS
- EfficientNet-Lite1
- MobileNetV2
- MobileNetV3-Small
- MobileNetV3-Large

All models were trained using transfer learning with ImageNet-pretrained weights.

**MobileViT-XXS** achieved the highest test classification performance, with an accuracy of approximately **0.973**.

However, model selection for embedded deployment also considered computational requirements, conversion compatibility, quantisation behaviour and robustness. Based on these criteria, **EfficientNet-Lite1** was selected for the final embedded implementation.

---

## Dataset

The multiclass dataset contains **4,961 grapevine leaf images** distributed across five classes:

| Class | Number of Images |
| --- | ---: |
| Healthy Leaf | 1000 |
| Black Rot | 970 |
| Esca | 1002 |
| Downy Mildew | 997 |
| Powdery Mildew | 991 |
| **Total** | **4961** |

The dataset combines images acquired under controlled-background and vineyard-background conditions.

Images were obtained from publicly available grapevine disease datasets, together with additional Black Rot images acquired at the vineyards of the **Escola Superior Agrária de Coimbra, Universidade Politécnica de Coimbra (ESAC-UPC)**.

For the multiclass experiments, the dataset was divided into:

- **70% training**
- **15% validation**
- **15% testing**

The complete dataset is not included directly in this repository. Information regarding the original datasets and image sources should be consulted before redistribution, as different datasets may be subject to different licenses and usage conditions.

---

## Robustness Analysis

A preliminary robustness analysis was performed on the lightweight convolutional models to investigate their sensitivity to controlled weight perturbations.

Uniform perturbations were applied to different groups of layers:

- Full model
- Early backbone layers
- Middle backbone layers
- Late backbone layers
- Depthwise convolutional layers
- Classifier head

The perturbations were applied directly to selected trained weights without retraining.

For each non-zero perturbation level, **100 independent experiments** were performed.

The perturbation magnitude was also analysed using the **weight-level signal-to-noise ratio (SNR)**.

The results showed that robustness depends both on:

- the neural network architecture;
- the region of the network affected by the perturbation.

Earlier backbone layers and depthwise convolutional layers generally showed greater sensitivity, while the classifier head was among the most robust regions.

---

## Embedded Deployment

**EfficientNet-Lite1** was selected for the embedded implementation.

The final system consists of:

- **Raspberry Pi 5**
- **Hailo-8L AI accelerator**
- **Raspberry Pi Camera Module 3**

The deployment pipeline consisted of:

1. Training the model in PyTorch
2. Exporting the trained model to ONNX
3. Converting the model using the Hailo Dataflow Compiler
4. Post-training quantisation
5. Compiling the model to Hailo Executable Format (`.hef`)
6. Running inference on the Hailo-8L accelerator
7. Acquiring images using the Raspberry Pi Camera Module 3

---

## Camera-Based Evaluation

The deployed model was evaluated under two different acquisition conditions.

### Controlled Camera-Based Testing

Grapevine leaf images were displayed on a monitor and captured using the Raspberry Pi Camera Module 3.

A total of **250 images** were evaluated, corresponding to 50 images per class.

The deployed EfficientNet-Lite1 model achieved an overall accuracy of:

**0.74**

### Vineyard Testing

The complete embedded system was subsequently evaluated using images acquired directly under vineyard conditions.

The captured images contained greater variability, including:

- natural illumination changes;
- complex vegetation backgrounds;
- different leaf orientations;
- variations in disease symptoms;
- pesticide residues and other natural surface variations.

The model predictions were compared with assessments independently provided by two experts.

The vineyard evaluation resulted in an average combined confidence-weighted agreement score of:

**0.436**

The experiment also highlighted the difficulty of identifying some grapevine diseases using visual symptoms alone, as disagreement was observed between the two expert assessments.

---

## Repository Content

The repository includes code related to the different stages of the project, including:

- Binary Black Rot classification
- Multiclass model training
- Model evaluation
- Plots generation
- Confusion matrix generation
- Robustness experiments
- Weight perturbation analysis
- ONNX model export
- Hailo model conversion and deployment
- Raspberry Pi camera acquisition
- Hailo-8L inference
- Controlled camera-based testing
- Vineyard evaluation

---

## Publications

This repository contains material associated with the Master's dissertation and related scientific work.

### Binary Black Rot Classification

**Selective Detection of Black Rot in Grapevine Leaves Using Convolutional Neural Networks**

Presented at the **15th International Symposium on Communication Systems, Networks and Digital Signal Processing (IEEE CSNDSP 2026)**.

### Multiclass Classification and Edge Deployment

**Camera-Based Grapevine Leaf Disease Classification Using Lightweight Deep Learning Models for Edge Deployment**

This work evaluates lightweight deep learning architectures for multiclass grapevine disease classification, robustness to weight perturbations and deployment on a Raspberry Pi 5 equipped with a Hailo-8L accelerator.

Submitted at **Smart Agricultural Technology**.

---

## Author

**Patrícia Sergueevna Moltchanova**  
University of Aveiro  
Instituto de Telecomunicações

### Supervision

- **Professor Luís Nero Alves**
- **Professor Carla Sofia Ferreira**

---

## Acknowledgements

The development of this work was supported by the **Instituto de Telecomunicações (IT)**.

The author would also like to thank **Engineer Filipe Melo** and **Engineer Sónia Leite** for their contribution to the evaluation and interpretation of the vineyard-acquired images. Additional thanks are given to Engineer Filipe Melo for his support during the vineyard visits and image acquisition.

---

## Data Availability

The complete image dataset is not directly included in this repository.

Some images originate from publicly available datasets and remain subject to the licenses and terms established by their original providers. Additional vineyard images were acquired during the development of this work.

Further information regarding dataset sources and availability can be provided separately.

---

## License

The source code in this repository is subject to the license provided in the repository.

The datasets and images used in this work may be subject to separate licenses and redistribution conditions.
The image datasets used in this work may be subject to separate licenses and terms defined by their original providers.
