# Digital-Collage-Creator (DCC)

DCC is a Graphics Design Software devloped using the language Python and the framework PyQt. It was created and tested on the Windows 10 platform, using Python v3.9.2 and PyQt v5. The application was created as current graphics design softwares either have high usability and low quality results or low usability and high quality results, DCC aims to fill this gap in the usability-quality tradeoff. This is done by designing the application around the creation of one type of digital art, **digital collage**, this allows many of the surplus tools and features to be stripped away while maintaining a familiar interface to other design softwares.

## Showcase Video

[3 minute video](https://www.youtube.com/watch?v=Pg2cWcYQs60)

## Features

- Insert/delete images
- Resize/rotate/drag layers
- Move layer up/down on Z-axis
- Toggle layer visibility
- Crop layers
- Cutout layers
    - Smooth path outline
    - Soften cutout edges
- Undo/redo actions
- Alter image:
    - Brightness
    - Sharpness/Blur
    - Contrast
    - RGB Levels
 - Add text layer
 - Generate colour palette from an image layer
    - From the palette generate a gradient background
 - Randomise properties of a layer
    - Position/scale/rotation/colour/filters
    - Lock in properties
 - Save canvas as .png image file
 - Save canvas as .dcc file format
    - JSON format to import project including layers and property values


## Installing and running DCC

### Python v3.9.2

[Download](https://www.python.org/downloads/release/python-392/)

### Additional Libraries

Libraries can be download using [pip](https://pip.pypa.io/en/stable/installing/), the standard package manager for Python. The following list contains the necessary libraries to install;

    pip install PyQt5
    pip install Pillow
    pip install matplotlib
    pip install numpy
    pip install sklearn
    pip install opencv-python
    pip install uuid
    pip install pathlib
    pip install shutil
    
The project also makes use of multiple in-built libraries: sys, math, os, collections, random, json, base64 and io.

### Running the program

The icon resources are provided for you to view, however, these have been pre-compiled and are automatically imported into the program from **resources.py**. To run the application execute the **Digital Collage Creator.py** file.

## Screenshots

<img src="https://raw.githubusercontent.com/RobertCooney99/Digital-Collage-Creator/main/images/DCC-Example-Image.png" width="400">
