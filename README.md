# Visual-Inspection


Pipeline to train a model that classifies items as 'good'/'defected'.
Trained withou any labels for defected region in the images, model is able to predict bounding box with defects for images classified as 'defected'. This is achieved by utilizing feature maps of the last convolutional layers.