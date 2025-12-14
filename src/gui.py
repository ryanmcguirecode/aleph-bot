import logging

import cv2
import numpy as np
from lerobot.processor import RobotObservation

log = logging.getLogger("GUI")


class Gui:

    def draw_scale_indicator(
        self, frame: np.ndarray, scale: float, corner: str = "top_right"
    ) -> np.ndarray:
        """
        Draws a larger, fully-filled motion-scale indicator box in a screen corner.
        The entire box is filled with a color representing the scale intensity.
        """
        h, w, _ = frame.shape

        # Determine color based on scale
        if scale < 0.3:
            color = (0, 255, 0)  # Green for low
        elif scale < 0.7:
            color = (0, 255, 255)  # Yellow for medium
        else:
            color = (0, 0, 255)  # Red for high

        # Box size (twice as large as before)
        bar_w, bar_h = 240, 40
        margin = 20

        # Positioning
        if corner == "top_right":
            x0, y0 = w - bar_w - margin, margin
        elif corner == "bottom_right":
            x0, y0 = w - bar_w - margin, h - bar_h - margin
        elif corner == "bottom_left":
            x0, y0 = margin, h - bar_h - margin
        else:
            x0, y0 = margin, margin

        # Draw background (darker border)
        cv2.rectangle(
            frame, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (40, 40, 40), -1
        )

        # Fill the entire box with the color
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), color, -1)

        # Outline
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 2)

        # Centered text
        label = f"{scale:.1f}x"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x0 + (bar_w - text_size[0]) // 2
        text_y = y0 + (bar_h + text_size[1]) // 2
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return frame

    def stack_frames(self, frames_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Combine multiple camera frames into a single mosaic frame."""

        valid_frames = [f for f in frames_dict.values() if isinstance(f, np.ndarray)]
        if not valid_frames:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        # Resize all to the same height for horizontal stacking
        target_height = min(f.shape[0] for f in valid_frames) // 3 * 2
        resized = [
            cv2.resize(f, (int(f.shape[1] * target_height / f.shape[0]), target_height))
            for f in valid_frames
        ]

        # Stack horizontally (if many, split into two rows)
        if len(resized) <= 3:
            combined = np.hstack(resized)
        else:
            half = (len(resized) + 1) // 2
            top = np.hstack(resized[:half])
            bottom = np.hstack(resized[half:])
            w = max(top.shape[1], bottom.shape[1])
            combined = np.vstack(
                [
                    cv2.copyMakeBorder(
                        top, 0, 0, 0, w - top.shape[1], cv2.BORDER_CONSTANT
                    ),
                    cv2.copyMakeBorder(
                        bottom, 0, 0, 0, w - bottom.shape[1], cv2.BORDER_CONSTANT
                    ),
                ]
            )

        return combined

    def display(self, observation: RobotObservation, motion_scale: float):

        frames = {}
        for cam_name, frame in observation.items():
            if isinstance(frame, np.ndarray):
                frames[cam_name] = frame

        if frames:
            combined = self.stack_frames(frames)
            # Convert RGB → BGR for OpenCV display
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            # Add scale indicator overlay
            combined_bgr = self.draw_scale_indicator(combined_bgr, motion_scale)

            cv2.imshow("Teleoperation Cameras", combined_bgr)

    def close(self):
        cv2.destroyAllWindows()

    def safe_close(self):
        try:
            self.close()
        except Exception:
            log.error("Failed to disconnect robot")
