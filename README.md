# Advex Format Conversion Tools

Convert COCO and YOLO format annotations to Advex Platform format. These utilities help prepare your data for use with the Advex Platform's General Pipeline.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Converting from COCO Format](#converting-from-coco-format)
  - [Converting from YOLO Format](#converting-from-yolo-format)
- [Input Requirements](#input-requirements)
- [Output Format](#output-format)
- [Common Issues](#common-issues)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/advex-conversion-tools.git
cd advex-conversion-tools
pip install -r requirements.txt
```

Required dependencies:
```txt
numpy
Pillow
opencv-python
pycocotools
tqdm
```

## Usage

### Converting from COCO Format

```bash
python convert_to_advex.py --format coco \
                          --input-dir /path/to/images \
                          --output-dir /path/to/output \
                          --coco-json /path/to/annotations.json
```

Expected COCO input structure:
```
input_directory/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations.json
```

### Converting from YOLO Format

```bash
python convert_to_advex.py --format yolo \
                          --input-dir /path/to/yolo_dataset \
                          --output-dir /path/to/output
```

Expected YOLO input structure:
```
yolo_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

## Input Requirements

- Images must be in `.jpg`, `.jpeg`, or `.png` format
- At least 10 images required
- Each image must have corresponding annotations
- COCO annotations must include either segmentation or bounding box data
- YOLO annotations must follow standard format: `<class> <x_center> <y_center> <width> <height>`

## Output Format

The conversion tools create the following structure required by Advex:

```
output_directory/
└── AdvexInputsFolder/
    ├── images/
    │   ├── img_1.png
    │   ├── img_2.png
    │   └── ...
    └── masks/
        ├── img_1.png
        ├── img_2.png
        └── ...
```

Requirements for output:
- All images are converted to PNG format
- Masks are binary (0 for background, 255 for foreground)
- Images and masks have matching filenames
- Minimum of 10 image-mask pairs


## Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **COCO JSON Not Found**
   - Ensure the path to your COCO JSON file is correct
   - Check that the JSON follows COCO format specifications

3. **YOLO Label Format Error**
   - Verify that label files contain normalized coordinates (0-1)
   - Check that each value is space-separated

4. **Invalid Mask Values**
   - Input annotations should result in binary masks (0 and 255 only)
   - Check source annotations for errors
