import string
from ultralytics import YOLO
import cv2
from skimage import io
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from sort.sort import *
import add_data
import visualise

ocr = PaddleOCR(use_angle_cls=True, lang='en')
mot_track = Sort()

# import easyocr
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#MAPPING FOR CHAR CONVERSIONS
char_to_int = {
    '0': 'O',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5'
}
int_to_char = {
    'O': '0',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S'
}

def lp_complies_format(text):
    if len(text) < 9:
        return False
    elif len(text) == 9:
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
            (text[2] in char_to_int.keys() or text[2] in ['0','1','2','3','4','5','6','7','8','9']) and \
            (text[3] in char_to_int.keys() or text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
            (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
            (text[5] in char_to_int.keys() or text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
            (text[6] in char_to_int.keys() or text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
            (text[7] in char_to_int.keys() or text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
            (text[8] in char_to_int.keys() or text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
            return True
    elif len(text) == 10:
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
            (text[2] in char_to_int.keys() or text[2] in ['0','1','2','3','4','5','6','7','8','9']) and \
            (text[3] in char_to_int.keys() or text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) and \
            (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in int_to_char.keys()) and \
            (text[6] in char_to_int.keys() or text[6] in ['0','1','2','3','4','5','6','7','8','9']) and \
            (text[7] in char_to_int.keys() or text[7] in ['0','1','2','3','4','5','6','7','8','9']) and \
            (text[8] in char_to_int.keys() or text[8] in ['0','1','2','3','4','5','6','7','8','9']) and \
            (text[9] in char_to_int.keys() or text[9] in ['0','1','2','3','4','5','6','7','8','9']):
            return True
    else:
        return False

def format_lp(text):
    lp = ''
    if (len(text) == 9):
        mapping = {
            0: int_to_char,
            1: int_to_char,
            2: char_to_int,
            3: char_to_int,
            4: int_to_char,
            5: char_to_int,
            6: char_to_int,
            7: char_to_int,
            8: char_to_int
        }
        for i in [0,1,2,3,4,5,6,7,8]:
            if text[i] in mapping[i].keys():
                lp += mapping[i][text[i]]
            else:
                lp += text[i]
    else:
        mapping = {
            0: int_to_char,
            1: int_to_char,
            2: char_to_int,
            3: char_to_int,
            4: int_to_char,
            5: int_to_char,
            6: char_to_int,
            7: char_to_int,
            8: char_to_int,
            9: char_to_int
        }
        try:
            for i in [0,1,2,3,4,5,6,7,8,9]:
                if text[i] in mapping[i].keys():
                    lp += mapping[i][text[i]]
                else:
                    lp += text[i]
        except:
            print("found some error in length of string detected")
    return lp

def pre_process_plate(cropped):
    grayscale_resize_test_license_plate = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(grayscale_resize_test_license_plate, 64, 255, cv2.THRESH_BINARY_INV)
    gaussian_blur_license_plate = cv2.GaussianBlur(license_plate_crop_thresh, (5, 5), 0)
    return gaussian_blur_license_plate

def readLp(edited_lp):
    result = ocr.ocr(edited_lp, cls=True)
    for line in result:
        if line != None:
            text = line[0][1][0]
            score = line[0][1][1]
            bbox = line[0][0]
            text = text.upper().replace(" ", "").replace("-", "")
            if lp_complies_format(text):
                return format_lp(text), score, bbox
            else:
                if score < 0.35:
                    return 'low_ConfS', score, bbox
                else:
                    return text, score, bbox
    return None, None, None

# load models
veh_model = YOLO('yolov8n.pt')
lp_detector = YOLO('./Model/best.pt')
print("models loaded")

def photo(img_path):
    image = io.imread(img_path)
    # detect license plate
    license_plates = lp_detector(image)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        lp_crop = image[int(y1): int(y2), int(x1):int(x2), :]

        lp_text, lp_score, bbox = readLp(pre_process_plate(lp_crop))

        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.imshow(lp_crop, cmap='gray')
        ax.text(30, 30, f'detected_text: {lp_text}', fontsize=12, color='white', ha='left', va='top')
        ax.text(30, 50, f'Score: {lp_score}', fontsize=12, color='white', ha='left', va='top')
        plt.show()

'''
for image- 
    load image *
    read image *
    detect license plates *
    crop lp *
    pre-process lp *
    read lp --
        complies format
        check for commonly mis recognized characters and correct them
    add bounding box to original image
    add read text to original image // write result
    add this to csv file - optional
    destroy any open window    
'''

def get_car(lp, track_ids):
    x1, y1, x2, y2, score, class_id = lp
    foundIt = False
    for j in range(len(track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt:
        return track_ids[car_indx]
    return -1, -1, -1, -1, -1

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

def video(vid_path):
    results = {}
    cap = cv2.VideoCapture(vid_path)
    frame_num = -1
    ret = True
    while ret and frame_num < 130:
        frame_num += 1
        print(frame_num)
        ret, frame = cap.read()
        if ret:
            results[frame_num] = {}
            detections =  veh_model(frame)[0]
            detections_ = []
            vehicles = [2, 3, 5, 6, 7]
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and detections_ == []:
                    detections_.append([x1, y1, x2, y2, score])
            print(detections_)
            # if detections_.shape[0] > 0:
            #     track_ids = mot_tracker.update(np.asarray(detections_))
            # else:
            #     pass
            track_ids = mot_track.update(np.asarray(detections_))
            license_plates = lp_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    lp_crop = frame[int(y1): int(y2), int(x1): int(x2), :]
                    lp_text, lp_score, bbox = readLp(pre_process_plate(lp_crop))
                    if lp_text is not None:
                        results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': lp_text,
                                                                        'bbox_score': score,
                                                                        'text_score': lp_score}}
    write_csv(results, './test.csv')

def camera():
    # cap = cv2.VideoCapture(0)
    #
    # if not cap.isOpened():
    #     print("can't open camera")
    #     exit()
    #
    # fram_num = -1
    # while cap.isOpened():
    #     fram_num += 1
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #
    #     license_plates = lp_detector(frame)[0]
    #     for license_plate in license_plates.boxes.data.tolist():
    #         x1, y1, x2, y2, score, class_id = license_plate
    #
    #         lp_crop = frame[int(y1): int(y2), int(x1): int(x2), :]
    #         lp_text, lp_score, bbox = readLp(pre_process_plate(lp_crop))
    #         print(lp_text, lp_score)
    #
    #
    #     # TO SHOW
    #     cv2.imshow('Capture feed', cv2.flip(frame, 1))
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    results = {}

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('Process_files/out.mp4', fourcc, fps, (width, height))

    frame_num = -1
    while cap.isOpened():
        frame_num += 1
        print(frame_num)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            results[frame_num] = {}
            detections = veh_model(frame)[0]
            detections_ = []
            vehicles = [2, 3, 5, 6, 7]
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and detections_ == []:
                    detections_.append([x1, y1, x2, y2, score])
            print(detections_)

            track_ids = mot_track.update(np.asarray(detections_))
            license_plates = lp_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    lp_crop = frame[int(y1): int(y2), int(x1): int(x2), :]
                    lp_text, lp_score, bbox = readLp(pre_process_plate(lp_crop))
                    if lp_text is not None:
                        results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': lp_text,
                                                                        'bbox_score': score,
                                                                        'text_score': lp_score}}
            cv2.imshow('Capture feed', cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    write_csv(results, './test.csv')
    add_data.use_in_final()
    visualise.visual('Process_files/out.mp4')

    cap.release()
    cv2.destroyAllWindows()

# load  photo* / video / camera-feed
# read frames --for video
# only for video and camera feed - detect vehicles
# track vehicles
# detect license plate*
# assign lp to a car
# crop lp - process lp
# read lp
# add to results
# write results
# destroy any open window