## Model installation, training and evaluation

### Installation
- Python version == 3.8.13

```
    conda create -n nlu python=3.8 -y
    conda activate nlu
    pip install -e . --upgrade --use-feature=in-tree-build
```

### Folder Tree
BKAI Folder Tree (Data Folder):
- BKAI
    + word-level
      ~ train
        + seq.in
        + label
        + seq.out
      ~ dev
        + seq.in
        + label
        + seq.out
      ~ test
        + seq.in
        + label (might not)
        + seq.out (might not)
      ~ intent_label.txt
      ~ slot_label.txt

### Augmentation
usage: augment_data.py [-h] [--dataset-path DATASET_PATH]
                       [--trainset TRAINSET]
                       [--train_intent_label TRAIN_INTENT_LABEL]
                       [--train_slot_label TRAIN_SLOT_LABEL]
                       [--valset VALSET]
                       [--val_intent_label VAL_INTENT_LABEL]
                       [--val_slot_label VAL_SLOT_LABEL]
                       [--intent-label-file INTENT_LABEL_FILE]
                       [--slot-label-file SLOT_LABEL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-path DATASET_PATH
                        path to the dataset
  --trainset TRAINSET   name of the training dataset file
  --train_intent_label TRAIN_INTENT_LABEL
                        name of the training intent label file
  --train_slot_label TRAIN_SLOT_LABEL
                        name of the training slot label file
  --valset VALSET       name of the validation dataset file
  --val_intent_label VAL_INTENT_LABEL
                        name of the validation intent label file
  --val_slot_label VAL_SLOT_LABEL
                        name of the validation slot label file
  --intent-label-file INTENT_LABEL_FILE
                        name of the intent label file
  --slot-label-file SLOT_LABEL_FILE
                        name of the slot label file
