# object-classificat-with-keras-and-mobilenetV2
Using this script, you can train any CNN structure exists on `Keras` such as `MobileNetV2`

 Object classification using `Keras` and specifically `MobileNetV2`

 *Usage*:
 Adjust the parameters `num_classes` , `SIZE_h`, `SIZE_w`, `train_batchsize` for your applications
 Define the `training`, `validation`, and `testing folders`. Each class images should be subfoldered like `0, 1, ..., num_classes-1`. 
 Define augmentation variables in Data Generators
 Adjust variables in `base_model = MobileNetV2` based on your needs or leave it as it is.

 *Run*:
 type in your command window: `python keras_object_classification.py`

 *P.S.* I omitted the folders for this demo code.
