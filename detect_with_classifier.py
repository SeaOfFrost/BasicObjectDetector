from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from detection_helpers import sliding_window
from detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

# Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-s", "--size", type = str, default = "(200, 150)", help = "ROI size")
ap.add_argument("-c", "--min-conf", type = float, default = 0.9, help = "Min. Probability to filter weak detections")
ap.add_argument("-v", "--visualize", type = int, default = 1, help = "Whether to show extra visualizations for debugging")
args = vars(ap.parse_args())

# Constants
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

# ResNet(Pretrained)
print("Loading ResNet50")
model = ResNet50(weights = "imagenet", include_top = True)

# Load the input image
orig = cv2.imread(args["image"])
orig = cv2.resize(orig, width = WIDTH)
(H, W) = orig.shape[:2]

# Initialize Image Pyramid 
pyramid = image_pyramid(orig, scale = PYR_SCALE, minSize = ROI_SIZE)

rois = []
locs = []

start = time.time()

for image in pyramid:
    # Find the scale factor
    scale = W / float(image.shape[1])
    
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # Scale to find the coordinates in the original image
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)

        # Resize ROI to send as input to ResNet
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # Save results
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        if args["visualize"] > 0:
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

end = time.time()
print("Looping over pyramid/windows took {:.5f} seconds".format(end - start))

# convert the ROIs to a NumPy array
rois = np.array(rois, dtype="float32")
# classify each of the proposals
print("Classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("Classifying ROIs took {:.5f} seconds".format(end - start))
# decode the predictions
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for (i, p) in enumerate(preds):
    # Get the prediction information
    (imagenetID, label, prob) = p[0]

    if prob >= args["min_conf"]:
        # Find the bounding box
        box = locs[i]

        # Get predictions for the label and add the box and prob to list
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

for label in labels.keys():
    # clone the original image so that we can draw on it
	print("Showing results for '{}'".format(label))
	clone = orig.copy()
	# loop over all bounding boxes for the label
	for (box, prob) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
	# Results before NMS
	cv2.imshow("Before", clone)
	
    # NMS
    clone = orig.copy()
    # extract the bounding boxes, prediction, apply NMS
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)
	# Repeat the above for the new boxes
	for (startX, startY, endX, endY) in boxes:
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	# show the output after apply non-maxima suppression
	cv2.imshow("After", clone)
	cv2.waitKey(0)


