**SpaMode: A general framework for deciphering spatial multi-omics using multimodal mixture of disentangled experts**

## Overview
Spatial multi-omic technologies enable simultaneous multi-omic profiling within native tissue context, offering unprecedented opportunities to study biological processes and disease. The increasing complexity of spatial multi-omics data demands a versatile model capable of addressing diverse scenarios, typically vertical (within a section), horizontal (across sections), and mosaic (across distinct omics) integration. 
Here, we propose SpaMode, a general framework that can be adopted to all the three types of spatial multi-omic circumstances. SpaMode disentangles each omics modality into omics-invariant and omics-variant distributions to characterize underlying biomolecular commonalities and specificities, and then hierarchically aggregates these distributions to resolve spatial heterogeneity. Horizontal integration and mosaic integration are unified within the SpaMode framework through multi-slice joint regularization and imputation of missing modalities. We benchmark SpaMode across vertical integration, horizontal integration, and mosaic integration, which demonstrate that SpaMode outperforms existing, targeted approaches in all integration settings. Furthermore, SpaMode provides novel insights into how invariant and variant multi-level biomolecular features contribute divergently to tissue spatial context, offering an interpretable alternative to black-box neural network. SpaMode provides a general and trustworthy solution for spatial multi-omic data analysis, paving the way for systematically decoding the complex mechanisms of cellular states and disease evolution in situ.

![Local Image](./Pic/Fig1_Framework.jpg)

## Tutorial
For the step-by-step tutorial, please refer to:
[https://spamode.readthedocs.io/en/latest/](https://spamode.readthedocs.io/en/latest/)