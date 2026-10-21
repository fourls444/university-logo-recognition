# University Logo Recognition

A Computer Vision coursework project that uses a convolutional neural network (CNN) to classify university logos. It prepares images with Pillow and NumPy, trains a TensorFlow/Keras model, and displays a university name and website from a CSV file.

Start with **[logo.ipynb](logo.ipynb)** to follow the work from image preparation to a prediction example. The task is image classification, not locating logos inside a photograph.

## Run the notebook

1. Open this project folder in VS Code with the Python and Jupyter extensions.
2. Create a separate environment from the project terminal:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Open `logo.ipynb` and select the `.venv` Python environment as the kernel. On another operating system, use that environment's Python executable instead.
4. Run the cells in order. The notebook finds its data when the working directory is this project or one of its subfolders.

**Before running the training cell, back up `model.h5` and `label.npy`. Training replaces both files.** To try an existing model without training, run the import and path cells, then the model-loading, prediction-function, and example cells. Skip model creation, training, and training-history charts.

`model.h5` is kept locally and excluded from Git because of its size. A fresh checkout does not include a pretrained model. You need a matching model and label file to run predictions. Read the limitations below before training or interpreting the predictions.

## Files

- `logo.ipynb`: the main walkthrough, training charts, and prediction example.
- `train.py`: an alternative training script. Run `python train.py` to train, not to test the installation.
- `test.py`: a manual prediction example. Run `python test.py` with a matching model and label file. This is not an automated test suite.
- `university_info.csv`: university names in English and Thai, plus website links.
- `Logo/`: training images grouped by university folder.
- `Image/`: example images used to try predictions. These are not an independent test set.
- `label.npy`: the saved mapping from class numbers to university names. Keep it paired with its model and load only trusted copies.

The notebook and scripts use RGB images resized to **256 × 256** pixels and scaled to the range 0–1. Resizing can stretch a non-square logo. A model trained with a different image size needs the matching preprocessing.

The notebook includes training-history charts. The training script uses all loaded images without a validation split, so it is not an evaluation substitute for the notebook.

## Current limitations

- There are 30 university folders with four training images each. This is a small coursework dataset.
- All 28 example images in `Image/` also occur in `Logo/`. Predictions on these images do not measure performance on unseen images.
- Class numbers are assigned from folder iteration order, but the saved names are built from CSV order. These orders can differ. Fix this mapping before trusting predicted university names or retraining for evaluation.
- Four folder names differ from their CSV names because of punctuation or spelling. These names need to be made consistent.
- The notebook uses a validation split, but there is no independent held-out test set or verified unseen-image accuracy.
- The model always chooses one of the known classes. It does not reliably reject an unrelated image.

Existing notebook outputs are saved examples, not a new verification run. Logo rights belong to their respective owners. This project does not imply endorsement by any university; verify image sources and redistribution permissions before sharing datasets.

## Next steps

Fix the label mapping and university names, prepare non-overlapping training and evaluation data, and report results on new images. A web upload demo can follow after those checks.
