import cv2
import albumentations as A
from pathlib import Path


input_folder = "nnUNet_raw/Dataset204_PenSeg_augmentation"
output_folder_name = "dataset_augmented"

START_INDEX = 84  

def create_output_folder_structure(output_folder):
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    for subfolder in ["imagesTr", "labelsTr"]:
        (output / subfolder).mkdir(parents=True, exist_ok=True)


transform = A.Compose([
    A.Rotate(
        limit=180,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.7
    ),

    A.RandomScale(
        scale_limit=0.15,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.5
    ),

    A.ElasticTransform(
        alpha=50,
        sigma=50,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=0.3
    ),

    A.PadIfNeeded(
        min_height=419,
        min_width=419,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    )
])



def data_augmentation(input_folder, output_folder, transform):

    input_path = Path(input_folder)

    images_path = input_path / "imagesTr"
    masks_path = input_path / "labelsTr"

    output = Path(output_folder)
    output_images = output / "imagesTr"
    output_masks = output / "labelsTr"

    image_files = sorted([f for f in images_path.iterdir() if f.is_file()])

    index = START_INDEX

    for image_file in image_files:

       
        base_name = image_file.stem.rsplit("_", 1)[0]
        mask_file = masks_path / f"{base_name}.png"

        if not mask_file.exists():
            print(f"Keine Maske gefunden für: {image_file.name}")
            continue

        image = cv2.imread(str(image_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Fehler beim Laden: {image_file.name}")
            continue

       
        augmented = transform(image=image, mask=mask)

        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        
        new_id = f"stift_{index:04d}"

        cv2.imwrite(str(output_images / f"{new_id}_0000.png"), aug_img)
        cv2.imwrite(str(output_masks / f"{new_id}.png"), aug_mask)

        index += 1


current_dir = Path(__file__).parent

input_path = current_dir / input_folder
output_path = current_dir / "nnUNet_raw" / "Dataset204_PenSeg_augmentation" / output_folder_name

create_output_folder_structure(output_path)

data_augmentation(input_path, output_path, transform)