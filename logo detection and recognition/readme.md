# Title

Logo detection and recognition using pytorch and google api

# Training Methodology

due to the poor and un-annnotated dataset, we follow an approach like the simi-supervised method. the method was creating dataset from the original logos only by taking the logos and apply it on another background images which are the background images of flicker-32 dataset. then train the dataset to classify logos. applying google api logo and text detection to get the locations that may contain the logo. then at each correct classification, add the classified logo new images to the dataset and augment it by rotation, shearing, contrast changes,cropping and zooming to inc the dataset with these new real logos. then fine tunening the model with last weights and train and so on.

the automatic created dataset also augmentd and the final dataset was 251987 , 160*160, images.

# the model architecture

1-the model is pretrained-resnet34 then train it to classify the logos.

2-the loss function used is "FocalLoss" to overcome the un-balanced classes images number due to the the new dataset added after each correct classification.

# results

the final testing results were about 99.99%

# notes

to get the best results try to pass an image contains an obvioused logo to make it able to detect.

# future work
 
as the model is trained on data most of them are created automatically by applying the original logos on the background ds not on real ones. if you passed noisy or blurred real one, it may work fine but not the best. so if you can collect more real data or do the same as "methodology" by applying it on lots of real data and with each correct classification add the image and its augmented ones to the dataset to train it after that, it will be better and better.

# make fast detection

to use the model in making fast detection by passing an image and get you all the logos in it. in "get the logos in image.py" file, use the function "getImageLogo" by passing it the client after making connection with api, imagePath ,device used (cuda or cpu), modelWeightPath="model_47.pth" (the model wights).

client is the return of "make connection function" like:- 
client = makeConnection(r'text and logo detection_credintials_avian-tract-283207-bc0721fc8622.json')

** after applying google api and returning the places that might contain the logos, I pass this part but increase 10 at the buttom right corner and decreasing 10 from top left corner then resizing this cropped part to 160*160 (as the images model trained on) and pass it to the model. you can change the passed top and buttom corners added and subtracted number but it's the best numbers, 10, we found depending on the real un-annotated DS.

this function will return you:-

1- list of lists containing the logos in the image and its places

** it may detect the same logo at the same place (approximately the same) so if you will plot the bounding boxes, use the IOU before it.

2- the logo response of google api itself

** take care, google response may be empty.