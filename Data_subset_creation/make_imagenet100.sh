#!/bin/bash

# === Configuration ===
SRC_ROOT=../../imagenet           # Path to full ImageNet directory
DST_ROOT=../../imagenet-100       # Output path for ImageNet-100 subset
NUM_CLASSES=100                   # Number of classes to sample
SELECTED_CLASSES=selected_classes.txt

# === Make directories ===
mkdir -p "$DST_ROOT/train"
mkdir -p "$DST_ROOT/val"

# === Randomly select class folders ===
cd "$SRC_ROOT/train" || exit 1
shuf -n "$NUM_CLASSES" -e */ | sed 's#/##' > "$DST_ROOT/$SELECTED_CLASSES"
echo "✅ Selected $NUM_CLASSES classes. List saved to $DST_ROOT/$SELECTED_CLASSES"

# === Copy folders ===
while read -r class; do
  echo "📦 Copying class: $class"
  cp -r "$SRC_ROOT/train/$class" "$DST_ROOT/train/"
  cp -r "$SRC_ROOT/val/$class" "$DST_ROOT/val/"
done < "$DST_ROOT/$SELECTED_CLASSES"

echo "✅ ImageNet-100 created at: $DST_ROOT"
echo "Press any key to exit..."
read -n 1 -s