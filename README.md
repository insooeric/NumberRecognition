# NumberRecognition

A number recognition AI using Convolutional Neural Network (CNN) built from scratch without using TensorFlow or any other pre-built frameworks. Training images, which include 48,000 images for training and 28,000 images for testing, can be found in [HERE](https://www.kaggle.com/datasets/scolianni/mnistasjpg)

# Installation

If you only would like to try out the application, download latest export in release page then follow [Usage](#usage)

1. Clone or download the project
2. Open the project with Visual Studio Code
3. Unzip train_images.zip and test_images.zip
   - Note that train_images contain subfolder 0 to 9 which contains approximately 4,000 images for each of the subfolder (48,000 images in total) and test_images contain unlabeled 28,000 images.
   - Unzipping these files may take quite amount of time.
4. After unzipping both folders, place them in the root directory. Make sure to name those folders as "train_images" (should have 0 to 9 subfolders) and "test_images" (should have 28,000 unlabeled images).
5. Follow along train_model.ipynb under learning folder.

# Application

Once you complete train_model.ipynb, you will have a trained model in HDF5 file, called trained_model_2.h5 (you may also use the one that already exists)

To compile and run the code locally, do one of the followings:

- Click run button at the top right in your editor.
- In the VS Code terminal, navigate to application directory then run the following code:

```
> & your/python/path/python.exe path/where/the/appPy/exists/app.py
```

either one will create small window.

## Usage

1. From the GUI windows, click "Import Model."
2. Select an HDF5 (\*.h5) file.
3. Once a canvas appears at the bottom of the screen, start drawing.
4. Click "Clear" button to erase the entire canvas.

# Releases

- v1.0.0

  - Initial application

- v1.0.1
  - Execution file with example model in HDF5

# Additional Note

- Expected script runtime: 40 minutes
  - (This may vary between PC's specification)
- sidenotes.ipynb includes problems I have faced, and how I managed to solve it.
- Running the entire script for training_model.ipynb may take long. Hence, in training_model.ipynb, press CTRL + F and search for "########## CHECK POINT ##########". This way, skip the training part that generates "trained_model.h5"
  - However, PLEASE at least read through the previous code since the comments explains what's really happening under the hood.
