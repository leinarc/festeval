# FESTEVAL (FUNSD-TESSERACT-LEVENSHTEIN-EVALUATION)

## Setup

Install Python (made on version 3.13.2)

Clone the repo and cd to project folder
```
git clone https://github.com/leinarc/festeval.git
cd festeval
```

Get the FUNSD dataset from here: https://guillaumejaume.github.io/FUNSD/

Extract the dataset to have a folder structure like this:
```
funsd
├── testing_data
│   ├── images
│   │   └── ... # .jpg files
│   └── annotations
│       └── ... # .json files
└── training_data
    ├── images
    │   └── ... # .jpg files
    └── annotations
        └── ... # .json files
eval.py
README.md
requirements.txt
```

## Run evaluation
```
# Install requirements
pip install -r requirements.txt
# Run evaluation
python eval.py
```