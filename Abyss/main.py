from pathlib import Path
from logzero import logger, logfile
from picamera import PiCamera
from orbit import ISS
from time import sleep
import datetime
from skyfield.api import load
import csv
import numpy as np
import cv2
from PIL import Image, ImageChops
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

def contrast_stretch(image):
    """Increase the contrast of a given image"""
    in_min = np.percentile(image, 5)
    in_max = np.percentile(image, 95)
    out_min = 0.0
    out_max = 255.0
    out = image - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min
    return out

def compute(file_path):
    """
    Function that compputes the albedo of a given image
    :param file_path: the path to the image
    :return: the albedo of the image
    """
    # we open the image and convert it to RGBA, adding max opacity (255) to all pixels
    image = Image.open(file_path)
    if image.mode == "RGB":
        image.putalpha(255)
    dir_path = Path(__file__).parent.resolve()
    lens = Image.open(dir_path / "lens_transparent3.png")
    
  # we subtract the lens from the image (the black zone from margins)
    image = ImageChops.subtract(image, lens)

    width, height = image.size
    pixels = image.load()

    # list of albedo values for each pixel
    albedos = []

    for i in range(width):
        for j in range(height):
            r, g, b, a = pixels[i, j]
            if a == 0:
                continue
            luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
            albedos.append(luminance / 255 * 0.65) # 0.65 is the albedo value of a white sheet of paper, and we compute
            # the albedo relative to this
 
    # we return the average of albedos of pixels
    return sum(albedos)/len(albedos)

def main(argv):
    """Create the outline of an image using Sobel"""
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    # Load the image
    src = cv2.imread(argv, cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    
    
    src = cv2.GaussianBlur(src, (3, 3), 0)
    
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals)
    e.g. 98° 34' 58.7 to "98/1,34/1,587/10"
 
    Return a tuple containing a boolean and the converted angle,
    with the boolean indicating if the angle is negative.
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    location = ISS.coordinates()
 
    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(location.latitude)
    west, exif_longitude = convert(location.longitude)
 
    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"
 
    # Capture the image
    camera.capture(image)

def create_csv_file(data_file):
    """Create a new CSV file and add the header row"""
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Counter", "Date/time", "Latitude", "Longitude")
        writer.writerow(header)
 
def create_result_file(data_file):
    """Create a new CSV file and add the header row"""
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Counter", "Average Albedo", "Has Water (EdgeTPU)", "Has Water (Albedo)")
        writer.writerow(header)
 
def add_csv_data(data_file, data):
    """Add a row of data to the data_file CSV"""
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        
def coral(image_file1):
    """Use the Coral TPU module to identify water in images"""
    script_dir1 = Path(__file__).parent.resolve()

    model_file1 = script_dir1/'models/astropi-water-vs-no_water.tflite' # name of model
    data_dir1 = script_dir1/'data'
    label_file1 = data_dir1/'water-vs-no_water.txt' # Name of your label file

    interpreter = make_interpreter(f"{model_file1}")
    interpreter.allocate_tensors()

    size1 = common.input_size(interpreter)
    image1 = Image.open(image_file1).convert('RGB').resize(size1, Image.Resampling.LANCZOS)

    common.set_input(interpreter, image1)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    labels = read_label_file(label_file1)
    for c in classes:
        return f'{labels.get(c.id, c.id)} {c.score:.5f}'
    
# Find the relative path of the main.py file and create the .csv result files    
base_folder = Path(__file__).parent.resolve()
data_file = base_folder/"data.csv"
result_file = base_folder/"results.csv"
create_csv_file(data_file)
create_result_file(result_file)

# Initialise the camera 
cam = PiCamera()
cam.resolution = (2592, 1944)

# Load the timescale of the ISS
ephemeris = load('/home/sandbox/de421.bsp')
timescale = load.timescale()
# Initialise the photo counter
counter = 0
computed_photo_count = 0 
# Record the start and current time
start_time = datetime.datetime.now()
now_time = datetime.datetime.now()
lens = Image.open(f"{base_folder}/lens_transparent3.png")
# Run a loop for (almost) three hours
while (now_time < start_time + datetime.timedelta(minutes=170)):
    try:
        if(ISS.at(timescale.now()).is_sunlit(ephemeris)):
            print("Day")
            location = ISS.coordinates()
            # Save the data to the file
            data = (
                counter,
                datetime.datetime.now(),
                location.latitude.degrees,
                location.longitude.degrees,
            )
            add_csv_data(data_file, data)
            # Capture image
            image_file = f"{base_folder}/photo_{counter:03d}.jpg"
            capture(cam, image_file)
            # Log event
            logger.info(f"iteration {counter}")
            counter += 1
            sleep(30)
            # Update the current time
            now_time = datetime.datetime.now()
        else:
            print("Night")
            if computed_photo_count < counter and counter != 0: # counter=cate poze am facut pana acum
                # Compute the data
                
                
                # Set the image to 255 alpha
                image = Image.open(f"{base_folder}/photo_{computed_photo_count:03d}.jpg")
                if image.mode == "RGB":
                    image.putalpha(255)
                if lens.mode == "RGB":
                    lens.putalpha(255)
                    
                # Crop the black outline of the image
                crop = ImageChops.subtract(image, lens)
                crop = np.array(crop)
                width, height = image.size
                
                # Increase the contrast of the image, find the average albedo and identify water
                contrasted = contrast_stretch(crop)
                cv2.imwrite(f"{base_folder}/photo_{computed_photo_count:03d}.jpg", contrasted)
                contour = main(f"{base_folder}/photo_{computed_photo_count:03d}.jpg")
                cv2.imwrite(f"{base_folder}/photo_contour_{computed_photo_count:03d}.jpg", contour)
                albedo=compute(f"{base_folder}/photo_{computed_photo_count:03d}.jpg")
                classified=coral(f"{base_folder}/photo_{computed_photo_count:03d}.jpg")
                if albedo<=0.075 and albedo>=0.05:
                    albedo_water='yes'
                else:
                    albedo_water='no'
                # Log info about the current photo
                logger.info(
                    "computed photo with index " + str(computed_photo_count) + " started at " + str(
                        now_time) + " and ended at " + str(datetime.datetime.now())
                )
 
                # Write data to the result file
                data = (
                    computed_photo_count,
                    albedo,
                    classified,
                    albedo_water
                )
                add_csv_data(result_file, data)
                computed_photo_count += 1
            # Update the current time
            now_time = datetime.datetime.now()
    except Exception as e:
        # Log any exeption and continue
        logger.error(f'{e._class.name_}: {e}') 

