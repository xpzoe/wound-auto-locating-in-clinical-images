# wound-auto-locating-in-clinical-images
Decubitus ulcer is a severe concern and can form in multiple locations on the body's surface. Prevention and treatment of decubitus ulcers are possible but challenge physicians and caregivers. This repository uses deep learning to localize decubitus ulcer wounds on the skin surface, offering the possibility of automatic diagnosis. 

# Dataset
A dataset of decubitus ulcer is created and proved valid. This dataset categorizes images into 11 classes indicating where the wound locates. Contact xpzoe522@outlook.com for more information.

A labeling GUI and a cropping GUI are provided. The later one can be used for refining raw images to center the interested content. Images should be gathered in folders according to categorise.

# Run
```
main.py
training.py
testing.py
my_models.py
prepare_data
    |--- Crop_GUI.py
    |--- Cropper.ui
    |--- Labeling_Gui.py
    |--- Labeling.ui
    |--- prepare_dataset.py
evaluate
    |--- evaluate.py
```
1. Prepare dataset by gathering images in category folders;
2. Execute main.py;
3. Test result will be saved in given paths.

In evaluate.py, functions are provided to evaluate testing performances and visualize results.

# Todo
* add regression part
