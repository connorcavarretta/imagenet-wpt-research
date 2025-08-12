# Efficient Deep Learning for Space-Based Image Recognition

This repository supports a graduate thesis investigating methods to improve the efficiency and robustness of deep learning models for image recognition in space exploration missions.  
The project centers on training and evaluating models using the ImageNet dataset, with an emphasis on techniques that reduce computational load and improve inference speed without sacrificing accuracy.  

A key area of experimentation is the integration of **Wavelet Packet Transform (WPT)** into the data loading pipeline, enabling frequency-domain feature extraction prior to model training. This approach aims to improve model performance and efficiency, particularly for deployment in **resource-constrained environments** such as spacecraft and planetary rovers.

Planned outcomes include:
- Comparative benchmarks between baseline and WPT-augmented pipelines.
- Analysis of trade-offs between accuracy, model complexity, and runtime efficiency.
- Insights into the applicability of frequency-domain preprocessing for space-based vision tasks.

## Acknowledgments

This project includes code from the [ConvNeXt model](https://github.com/facebookresearch/ConvNeXt)
developed by Meta Platforms, Inc., and licensed under the MIT License.

The ConvNeXt source files are located in `models/convnext.py`, and the corresponding license
is available in `third_party/convnext/LICENSE`.
