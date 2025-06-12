import ast
import copy
import threading

import cv2
import cv_bridge
import numpy as np
import numpy.typing as npt
import rclpy
import rclpy.callback_groups
import rclpy.node
import sensor_msgs.msg
import std_msgs.msg
import torch
from PIDNet.models import pidnet


class ObjectSegmenter(rclpy.node.Node):
    MODEL_INPUT_SIZE = (1024, 512)

    class Result:
        def __init__(
            self,
            labels: npt.NDArray[np.uint8],
            source_image: npt.NDArray[np.uint8],
        ):
            self.labels = labels
            self.source_image = source_image

    def __init__(self):
        super().__init__("object_segmenter")

        checkpoint_file = (
            self.declare_parameter("checkpoint_file", "")
            .get_parameter_value()
            .string_value
        )
        num_of_classes = (
            self.declare_parameter("num_of_classes", 4)
            .get_parameter_value()
            .integer_value
        )
        self.class_colors = np.array(
            ast.literal_eval(
                self.declare_parameter(
                    "class_colors",
                    "[[0, 0, 0], [255, 255, 0], [0, 0, 255], [255, 0, 0]]",
                )
                .get_parameter_value()
                .string_value
            ),
            dtype=np.uint8,
        )
        self.standardization_mean = (
            self.declare_parameter("standardization.mean", [0.485, 0.456, 0.406])
            .get_parameter_value()
            .double_array_value
        )
        self.standardization_std = (
            self.declare_parameter("standardization.std", [0.229, 0.224, 0.225])
            .get_parameter_value()
            .double_array_value
        )
        self.use_compressed_image = (
            self.declare_parameter("use_compressed_image", True)
            .get_parameter_value()
            .bool_value
        )
        result_visualization_rate = (
            self.declare_parameter("result_visualization_rate", 10.0)
            .get_parameter_value()
            .double_value
        )

        self.model = pidnet.get_pred_model("pidnet-s", num_classes=num_of_classes)
        checkpoint: dict = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model_dict = self.model.state_dict()
        checkpoint = {
            k[6:]: v
            for k, v in checkpoint.items()
            if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
        }
        model_dict.update(checkpoint)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.cuda().eval()

        self.cv_bridge = cv_bridge.CvBridge()

        self.result_lock = threading.Lock()
        self.result: ObjectSegmenter.Result = None

        self.label_image_pub = self.create_publisher(
            sensor_msgs.msg.Image,
            "~/label_image",
            1,
        )

        if self.use_compressed_image:
            self.image_raw_sub = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                "image_raw/compressed",
                self.image_raw_compressed_callback,
                1,
            )
        else:
            self.image_raw_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                "image_raw",
                self.image_raw_callback,
                1,
            )

        self.visualize_result_timer = self.create_timer(
            1 / result_visualization_rate,
            self.visualize_result_callback,
            rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

    def image_raw_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage):
        source_image = self.cv_bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="rgb8"
        )
        self.segment_objects(source_image, msg.header)

    def image_raw_callback(self, msg: sensor_msgs.msg.Image):
        source_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.segment_objects(source_image, msg.header)

    def visualize_result_callback(self):
        if self.result is None:
            return

        with self.result_lock:
            result = copy.deepcopy(self.result)

        result_image = cv2.cvtColor(
            cv2.addWeighted(
                src1=self.class_colors[result.labels],
                src2=result.source_image,
                alpha=0.5,
                beta=1.0,
                gamma=0.0,
            ),
            cv2.COLOR_RGB2BGR,
        )

        cv2.imshow("Object segmenter", result_image)
        cv2.waitKey(1)

    def segment_objects(
        self, source_image: npt.NDArray[np.uint8], header: std_msgs.msg.Header
    ):
        source_image = cv2.resize(source_image, ObjectSegmenter.MODEL_INPUT_SIZE)

        input_image: npt.NDArray[np.float64] = source_image / 255
        input_image = (
            input_image - self.standardization_mean
        ) / self.standardization_std
        input_image = input_image.transpose((2, 0, 1))

        input_tensor = torch.from_numpy(input_image).float().unsqueeze(0).cuda()

        with torch.no_grad():
            prediction: torch.Tensor = self.model(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction,
            size=input_tensor.size()[-2:],
            mode="bilinear",
            align_corners=True,
        )
        labels = prediction.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        label_image_msg = self.cv_bridge.cv2_to_imgmsg(
            labels, encoding="8UC1", header=header
        )
        self.label_image_pub.publish(label_image_msg)

        with self.result_lock:
            self.result = ObjectSegmenter.Result(labels, source_image)


def main(args=None):
    rclpy.init(args=args)

    object_segmenter = ObjectSegmenter()

    rclpy.spin(object_segmenter)

    object_segmenter.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
