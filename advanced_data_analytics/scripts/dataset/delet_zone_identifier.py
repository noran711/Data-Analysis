import os

folders = [
    r"nnUNet_raw/Dataset204_PenSeg_augmentation/imagesTr",
    r"nnUNet_raw/Dataset204_PenSeg_augmentation/labelsTr",
    r"nnUNet_raw/Dataset204_PenSeg_augmentation/imagesTs"
]

for folder in folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Zone.Identifier Alternate Data Stream
            ads_path = file_path + ":Zone.Identifier"

            try:
                if os.path.exists(ads_path):
                    os.remove(ads_path)
                    print(f"Gelöscht: {ads_path}")
            except Exception as e:
                print(f"Fehler bei {ads_path}: {e}")