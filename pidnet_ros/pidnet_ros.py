import cv2
import cv_bridge
import numpy as np
import numpy.typing as npt
import rclpy
import rclpy.node
import sensor_msgs.msg
import torch
from PIDNet.models import pidnet


class PIDNetROS(rclpy.node.Node):
    INPUT_SIZE = (1024, 512)
    CLASS_COLORS = np.array(
        [
            [0, 0, 0],  # others: black
            [255, 255, 0],  # straight: yellow
            [0, 0, 255],  # fork: blue
            [255, 0, 0],  # grass: red
        ],
        dtype=np.uint8,
    )

    def __init__(self):
        super().__init__("pidnet_ros")

        checkpoint_file = (
            self.declare_parameter("checkpoint_file", "")
            .get_parameter_value()
            .string_value
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

        self.model = pidnet.get_pred_model("pidnet-s", num_classes=4)
        checkpoint: dict = torch.load(checkpoint_file, map_location="cpu")
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

        self.result_image_pub = self.create_publisher(
            sensor_msgs.msg.Image,
            "~/result_image",
            1,
        )

        self.image_raw_sub = self.create_subscription(
            sensor_msgs.msg.Image,
            "image_raw",
            self.image_raw_callback,
            1,
        )

    def image_raw_callback(self, msg: sensor_msgs.msg.Image):
        source_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        source_image = cv2.resize(source_image, PIDNetROS.INPUT_SIZE)

        input_image: npt.NDArray[np.float64] = source_image / 255
        # input_image = (
        #     input_image - input_image.mean(axis=0, keepdims=True)
        # ) / input_image.std(axis=0, keepdims=True, ddof=1)
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
        labels = prediction.argmax(dim=1).squeeze(0).cpu().numpy()

        result_image = cv2.addWeighted(
            src1=PIDNetROS.CLASS_COLORS[labels],
            src2=source_image,
            alpha=0.5,
            beta=1.0,
            gamma=0.0,
        )
        result_image_msg = self.cv_bridge.cv2_to_imgmsg(
            result_image, encoding="rgb8", header=msg.header
        )
        self.result_image_pub.publish(result_image_msg)


def main(args=None):
    rclpy.init(args=args)

    pidnet_ros = PIDNetROS()

    rclpy.spin(pidnet_ros)

    pidnet_ros.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
