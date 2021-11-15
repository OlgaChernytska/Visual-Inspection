# Visual-Inspection


Pipeline to train a model that classifies items as 'good'/'defective'.
Trained without any labels for defected region in the images, model is able to predict bounding box with defects for images classified as 'defective'. This is achieved by utilizing feature maps of the last convolutional layers.


## Arthitecture

**Training:**
VGG16 feature extractor pre-trained on ImageNet, classification head - Average Global Pooling and a Dense layer. Model outputs 2-dimensional vector that contains probabilities for class 'good' and class 'defective'. Finetuned only last 3 convolutional layers and a dense layer.


**Inference:**
During inference molde output probabilities as well as the heatmap. Heatmap is the linear combination of feature maps from layer conv5-3 weighted by weights of the last dense layer, and unsampled to match image size. From the dense layer, we take only weights that were used to calculate the score for class 'defective'. For each input image, model returns a single heatmap. High values in the heatmap correspond to pixels that are very important for a model to decide that this particular image is defective. This means, that high values in the heatmap show the actual location of the defect. Heatmaps are processed further to return bounding boxes of the areas with defects.


## Project Structure

- ```Training.ipynb``` - notebook with training, evaluation and visualization
- ```utils.py``` - functions used in the notebook
- ```model.h5``` - trained model 