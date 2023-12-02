# predict-compression
Predicting MSCC from T2w-MRI images of spinal cord

## Run locally
python cord_sense/main.py --dataset_dir data-multi-subject (not working yet)

You will first need to download the dataset from : https://github.com/spine-generic/data-multi-subject (follow the instructions)
and specify the path to data-multi-subject directory on your machine (required).

You will also need to install the sct_venv environment and run the command inside it. Follow the instructions here : https://spinalcordtoolbox.com/user_section/installation.html

## Acknowledgment
I would like to acknowledge the work of Jan Valosek and collaborators (https://github.com/spinalcordtoolbox/detect-compression/tree/main) which provided inspiration for this one. This work extensively uses the methods from SpinalCordToolbox for processing patient MRI images of the spinal cord as well.