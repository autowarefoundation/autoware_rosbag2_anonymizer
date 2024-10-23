import os

import cv_bridge

import supervision as sv
import cv2

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter

from autoware_rosbag2_anonymizer.model.unified_language_model import (
    UnifiedLanguageModel,
)
from autoware_rosbag2_anonymizer.model.sam2 import SAM2

from autoware_rosbag2_anonymizer.common import (
    create_classes,
    blur_detections,
    get_file_paths,
)


def anonymize_with_unified_model(config_data, json_data, device) -> None:
    # Segment-Anything-2 parameters
    SAM2_MODEL_CFG = config_data["segment_anything_2"]["model_cfg"]
    SAM_CHECKPOINT_PATH = config_data["segment_anything_2"]["checkpoint_path"]

    # Segment-Anything-2
    sam2 = SAM2(SAM2_MODEL_CFG, SAM_CHECKPOINT_PATH, device)

    unified_language_model = UnifiedLanguageModel(config_data, json_data, device)

    # Get rosbag paths from input folder
    rosbag2_paths = get_file_paths(
        config_data["rosbag"]["input_bags_folder"], [".db3", ".mcap"]
    )

    # Create output folder if it is not exist
    os.makedirs(config_data["rosbag"]["output_bags_folder"], exist_ok=True)

    for rosbag2_path in rosbag2_paths:
        reader = RosbagReader(rosbag2_path, 1)
        writer = RosbagWriter(
            os.path.join(config_data["rosbag"]["output_bags_folder"], rosbag2_path.split("/")[-1].split(".")[0]),
            config_data["rosbag"]["output_save_compressed_image"],
            config_data["rosbag"]["output_storage_id"],
        )

        for i, (msg, is_image) in enumerate(reader):
            if not is_image:
                writer.write_any(msg.data, msg.type, msg.topic, msg.timestamp)
            else:
                # Convert image msg to cv.Mat
                if "Compressed" in msg.type:
                    image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)
                else:
                    image = cv_bridge.CvBridge().imgmsg_to_cv2(msg.data)

                # Find bounding boxes with Unified Model
                detections = unified_language_model(image)

                # Run SAM2
                detections = sam2(image=image, detections=detections)

                # Blur detections
                output = blur_detections(
                    image,
                    detections,
                    config_data["blur"]["kernel_size"],
                    config_data["blur"]["sigma_x"],
                )

                # Print detections: how many objects are detected in each class
                print("\nDetections:")
                for class_id in range(len(unified_language_model.detection_classes)):
                    print(
                        f"{unified_language_model.detection_classes[class_id]}: {len([d for d in detections if d[3] == class_id])}"
                    )

                # Write blured image to rosbag
                writer.write_image(output, msg.topic, msg.timestamp)

                # Debug ------------------
                # DETECTION_CLASSES, CLASSES, CLASS_MAP = create_classes(json_data=json_data)

                # bounding_box_annotator = sv.BoundingBoxAnnotator()
                # annotated_image = bounding_box_annotator.annotate(
                #     scene=output,
                #     detections=detections,
                # )

                # labels = [
                #     f"{DETECTION_CLASSES[class_id]} {confidence:0.2f}"
                #     for _, _, confidence, class_id, _, _ in detections
                # ]
                # label_annotator = sv.LabelAnnotator()
                # annotated_image = label_annotator.annotate(
                #     output,
                #     detections,
                #     labels,
                # )

                # height, width = image.shape[:2]
                # annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
                # cv2.imshow("test", annotated_image)
                # cv2.waitKey(1)
                # Debug ------------------
