# VGG Face Gender Classifer

A gender classifier on portraits with features extracted by the [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) model.

## Architecture and training process

The final layer of VGG-Face was replaced by a two-layer MLP of 128 and 1 units with a final sigmoid activation. During transfer learning, weights of the base VGG-Face model are frozen, and only the added classification layer is tuned. 

Pretrained VGG-Face model in [Caffe implementation](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) was converted to a Keras model (`saved_models/VGG_FACE/VGG_FACE.h5` using the [MMDNN](https://github.com/microsoft/MMdnn) tool. The modified structure and trained model is saved as a TF `SavedModel` in `saved_models/VGG_FACE_gender`.

The total of 29437 images in the `data/aligned` folder is used for training, with 20% reserved for validation. Images in the `data/valid` folder are not touched during training, which the model with test on as shown in the `test.ipynb` notebook.

Convergence is observed at around 5 epochs. Deeper classifier layers were also experimented, though there was no performance gain in accuracy.

## Usage

The `train.py` script includes both training and evaluation processes. A list of arguments is the following:
- `-e`, `--epochs`: the number of epochs for training.
- `-b`, `--batch-size`: image batch size, default 32.
- `--eval-only`: run only evaluation.

For example, the following command runs training for 5 epochs and validation every epoch.
```
python train.py -e 5
```
If only evaluation is need on trained model, run
```
python train.py --eval-only
```

## Results

After 5 epochs, binary claasification accuracy with 0.5 threshold on the validation set is 0.9604 (saved model provided). Please see the `test.ipynb` notebook for accuracy demonstration, AUC value, visualization and accuracy distribution across age groups.