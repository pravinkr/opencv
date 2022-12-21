# stereo-depth-map
Stereo depth map


# Refer [this link](!https://plainenglish.io/blog/the-depth-i-stereo-calibration-and-rectification-24da7b0fb1e0) for reference code


## Script Details

1. Capturing checkerboard images using live video for individual cameras.

```
# Ensure the checkerboard size is defined. Default is (6,8) for checkerboard image Checkerboard-A4-25mm-8x6.pdf
python calibrate_camera_image_capture.py
```

2. Capturing chekerboard images using live vide for Stereo Cameras.
   ```
    # Change the video source
    # once you see the checkerboard image being correctly identified, capture image by pressing button p 

   python stereo_image_capture_for_calibration.py

   ```

3. Individual Camera Calibration

```
# Default checkerboard size in script is (6,8) - 6 rows and 8 columns
# Ensure at least 10-12 cheaker board images are captured from the camera at different angles.
# calibrate_camera_image_capture.py can be used to capture images

python calibrate.py
```

```camera_calibration.py```
This python script can be used to calibrate stereo camera. Ensure the image captured by stereo cameras are present on corresponding directories. Refer below sample command to run the script and generate calibration parameters.

```
python camera_calibration.py --image_dir="./left_camera" --prefix="imageL_" --image_format="png" --square_size=0.025 --width=8 --height=6 --save_file="Calibration_MatrixL.yaml"

Left Calibration is finished. RMS:  0.13236679976445279
Right Calibration is finished. RMS:  0.13381760158528713

```

```extract_calibration.py```
Script to extract the calibration matrix from saved calibration file.
```
extract_calibration("Calibration_MatrixL.yaml")
extract_calibration("Calibration_MatrixR.yaml")
```


4. Stereo Calibration
   ```
    # Use this script to generate calibration matrix for stereo camera setup.
    # The calibration parameter will be stored in yaml files.
    # These parameters will be read in future script to generate the depth map.
    # The checkerboard square size needs to be measured using scale and put in meters in square_size parameter

    python stereo_calibration.py --left_dir="./left_camera" --left_file="Calibration_MatrixL.yaml" --left_prefix="imageL_" --right_dir="./right_camera" --right_file="Calibration_MatrixR.yaml" --right_prefix="imageR_" --image_format="png" --square_size=0.025 --width=8 --height=6 --save_file="Calibration_MatrixStereo.yaml"

    Stereo calibration rms:  0.42216018533612254
 
   ```

5. Depth Map generation
   ```

    # For real time video stream
    python stereo_depth.py --calibration_file="Calibration_MatrixStereo.yaml" --left_source="/dev/video2" --right_source="/dev/video4" --is_real_time=True


   ```