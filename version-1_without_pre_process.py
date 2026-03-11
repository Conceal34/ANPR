from ultralytics import YOLO
import easyocr
import skimage as ski
from skimage import io
from skimage.color import rgb2gray
import skimage.filters
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'], gpu = False)

def readLp(edited_lp):
    detections = reader.readtext(edited_lp)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        return text, score
    return None, None

# for photo only 1st
# load models
veh_model = YOLO('yolov8n.pt')
lp_detector = YOLO('./Model/best.pt')
print("models loaded")

img_path = "test-files/sct_img.jpg"
image = io.imread(img_path)

car_id = 1
# detect license plate
license_plates = lp_detector(image)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate
    # assign lp to a car --for multiple cars (currently doing only 1 car)
    # crop lp
    lp_crop = image[int(y1 + 10): int(y2), int(x1 - 10):int(x2), :]

    plt.imshow(lp_crop)
    plt.show()

# load video / photo / camera-feed
# read frames --for video
# only for video and camera feed - detect vehicles
# track vehicles
# detect license plate
# assign lp to a car
# crop lp - process lp
# read lp
# add to results
# write results
# destroy any open window



# grayscale = rgb2gray(lp_crop)
# gaussian = ski.filters.gaussian(lp_crop, sigma=1, truncate=2.0)
# threshold_value = threshold_otsu(gaussian)
# bin_img = gaussian > threshold_value
# cleaned = closing(bin_img, square(3))
