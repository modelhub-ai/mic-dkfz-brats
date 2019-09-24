# mic-dkfz-brats
This repository hosts the contributor source files for the mic-dkfz-brats model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).

### Info 
This model needs a GPU to run. Please follow the quickstart instructions for GPU on our website to find out how to set up your system: [Quickstart Docs](https://modelhub.readthedocs.io/en/latest/quickstart.html)

## meta
| | |
|-|-|
| id | 14e79015-ae1d-49b7-9673-032f6e441d3d | 
| application_area | Medical Imaging, Segmentation | 
| task | Brain Tumor Segmentation | 
| task_extended | Brain tumor segmentation for the BraTS 18 challenge | 
| data_type | Nifti-1 volumes | 
| data_source | www.braintumorsegmentation.org | 
## publication
| | |
|-|-|
| title | No new-net | 
| source | International MICCAI Brainlesion Workshop | 
| url | https://link.springer.com/chapter/10.1007/978-3-030-11726-9_21 | 
| year | 2018 | 
| authors | Fabian Isensee,Philipp Kickingereder, Wolfgang Wick, Martin Bendszus, Klaus H. Maier-Hein | 
| abstract | In this paper we demonstrate the effectiveness of a well trained U-Net in the context of the BraTS 2018 challenge. This endeavour is particularly interesting given that researchers are currently besting each other with architectural modifications that are intended to improve the segmentation performance. We instead focus on the training process arguing that a well trained U-Net is hard to beat. Our baseline U-Net, which has only minor modifications and is trained with a large patch size and a Dice loss function indeed achieved competitive Dice scores on the BraTS2018 validation data. By incorporating additional measures such as region based training, additional training data, a simple postprocessing technique and a combination of loss functions, we obtain Dice scores of 77.88, 87.81 and 80.62, and Hausdorff Distances (95th percentile) of 2.90, 6.03 and 5.08 for the enhancing tumor, whole tumor and tumor core, respectively on the test data. This setup achieved rank two in BraTS2018, with more than 60 teams participating in the challenge. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cluster=10467106438092249798 | 
| bibtex | @inproceedings{isensee2018no, title={No new-net},author={Isensee, Fabian and Kickingereder, Philipp and Wick, Wolfgang and Bendszus, Martin and Maier-Hein, Klaus H},booktitle={International MICCAI Brainlesion Workshop},pages={234--244},year={2018},organization={Springer} | 
## model
| | |
|-|-|
| description | nnU-Net | 
| provenance |  | 
| architecture | CNN | 
| learning_type | Supervised | 
| format | .model | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).
