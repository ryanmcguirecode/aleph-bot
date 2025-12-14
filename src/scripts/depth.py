#!/usr/bin/env python3
"""
Display synchronized Infrared, Depth, and RGB streams from an Intel RealSense camera
at the highest supported depth resolution (1280x720 @ 5 Hz for most D4xx/D405 series).

Requirements:
    pip install pyrealsense2 opencv-python numpy
"""

import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    # --- Configure RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Highest common resolution among Depth, IR, and RGB streams
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)
    config.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8, 5)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)

    # Start streaming
    pipeline.start(config)
    print("[INFO] Streaming 1280x720 (IR + Depth + RGB) @ 5 Hz. Press ESC to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not ir_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Depth colormap for visualization
            depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

            # Convert grayscale IR to 3-channel BGR for stacking
            ir_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            # Combine side-by-side
            combined = np.hstack((ir_bgr, depth_colormap, color_image))

            # Add simple labels
            cv2.putText(combined, "Infrared", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(combined, "Depth", (460, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(combined, "RGB", (920, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            # Show
            cv2.imshow("Infrared | Depth | RGB (1280x720)", combined)
            if cv2.waitKey(1) == 27:  # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stream stopped.")


if __name__ == "__main__":
    main()
