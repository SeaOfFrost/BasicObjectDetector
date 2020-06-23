from skimage.transform import pyramid_gaussian

def sliding_window(image, step, ws):
    # slide over the image and return the window 
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image, scale = 1.5, minSize = (224, 224)):
    # Return the original image 
    yield image
    # Using scikit learn
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=1.5)):
        # if the image is too small, break from the loop
        if resized.shape[0] < minSize[1] or resized.shape[1] < minSize[0]:
            break
        # Return rescaled image
	    yield resized