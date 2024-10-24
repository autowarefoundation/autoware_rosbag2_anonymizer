import os
import yaml

import cv2
import cv_bridge

from supervision.dataset.formats.yolo import (
    detections_to_yolo_annotations,
    save_text_file,
)

from autoware_rosbag2_anonymizer.common import (
    create_classes,
    get_file_paths,
)

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader

from autoware_rosbag2_anonymizer.model.unified_language_model import (
    UnifiedLanguageModel,
)


def yolo_create_dataset(config_data, json_data, device) -> None:
    # Define classes
    DETECTION_CLASSES, CLASSES, CLASS_MAP = create_classes(json_data=json_data)

    unified_language_model = UnifiedLanguageModel(config_data, json_data, device)

    # Get ROS2 bag file names
    rosbag2_paths = get_file_paths(
        config_data["rosbag"]["input_bags_folder"], [".db3", ".mcap"]
    )

    # Dataset folders and yaml file paths
    DATASET_DIR_PATH = config_data["dataset"]["output_dataset_folder"]
    ANNOTATIONS_DIRECTORY_PATH = os.path.join(DATASET_DIR_PATH, "annotations")
    IMAGES_DIRECTORY_PATH = os.path.join(DATASET_DIR_PATH, "images")
    DATA_YAML_PATH = os.path.join(DATASET_DIR_PATH, "data.yaml")

    # Create folders if it is not exist
    os.makedirs(DATASET_DIR_PATH, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIRECTORY_PATH, exist_ok=True)
    os.makedirs(IMAGES_DIRECTORY_PATH, exist_ok=True)

    # Declare counter variable for naming
    IMAGE_COUNTER = 0
    SUBSAMPLE_COEFFICIENT = config_data["dataset"][
        "output_dataset_subsample_coefficient"
    ]

    for rosbag2_path in rosbag2_paths:
        reader = RosbagReader(rosbag2_path, SUBSAMPLE_COEFFICIENT)

        for i, (msg, is_image) in enumerate(reader):
            if not is_image:
                continue

            # Convert image msg to cv.Mat
            if "Compressed" in msg.type:
                image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)
            else:
                image = cv_bridge.CvBridge().imgmsg_to_cv2(msg.data)

            # Find bounding boxes with Unified Model
            detections = unified_language_model(image)

            # Print detections: how many objects are detected in each class
            print("\nDetections:")
            for class_id in range(len(unified_language_model.detection_classes)):
                print(
                    f"{unified_language_model.detection_classes[class_id]}: {len([d for d in detections if d[3] == class_id])}"
                )

            cv2.imwrite(
                filename=f"{IMAGES_DIRECTORY_PATH}/image{IMAGE_COUNTER}.jpg",
                img=image,
            )

            lines = detections_to_yolo_annotations(
                detections=detections,
                image_shape=image.shape,
                min_image_area_percentage=0.0,
                max_image_area_percentage=1.0,
                approximation_percentage=0.0,
            )
            save_text_file(
                lines=lines,
                file_path=f"{ANNOTATIONS_DIRECTORY_PATH}/image{IMAGE_COUNTER}.txt",
            )

            IMAGE_COUNTER += 1

            # box_annotator = sv.BoxAnnotator()
            # labels = [
            #     f"{DETECTION_CLASSES[class_id]} {confidence:0.2f}"
            #     for _, _, confidence, class_id, _, _ in detections
            # ]
            # annotated_image = box_annotator.annotate(
            #     scene=image, detections=detections, labels=labels
            # )

            # height, width = image.shape[:2]
            # image = cv2.resize(image, (width // 2, height // 2))

            # cv2.imshow("debug", image)
            # cv2.waitKey(1)

    data = {"names": DETECTION_CLASSES, "nc": len(DETECTION_CLASSES)}
    with open(DATA_YAML_PATH, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
