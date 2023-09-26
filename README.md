# GroundingDINO BBox Labeller
A bbox labeller built on [IDEA-Research GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## Directory Structure
```
weights/
    <groundingdino models goes here>
.gitignore
main.py
README.md
requirements.txt
```

## Download Model Checkpoint
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## Run Script
```
$ python main.py -i <input_image_dir> -o <output_dir> -t <prompt>
```