# run this to mount your google drive
from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np

def get_video_properties(video_path):
    """Gets the width, height, and FPS of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def resize_frame(frame, target_width, target_height):
    """Resizes a frame to target dimensions."""
    return cv2.resize(frame, (target_width, target_height))

# MoveNet integration
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from google.colab.patches import cv2_imshow
import json

def load_movenet_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

    return model.signatures["serving_default"]

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (192, 192))
    input_tensor = tf.convert_to_tensor(resized, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    return input_tensor

def get_keypoints_from_image(image, model):
    """
    Processes an image using MoveNet and returns keypoints with confidence.
    Returns a list of lists/tuples: [[x1, y1, conf1], [x2, y2, conf2], ...]
    """
    input_tensor = preprocess_image(image)
    outputs = model(input_tensor)
    keypoints_with_scores = outputs['output_0'].numpy()  # shape: (1, 1, 17, 3)
    # Flatten to 17x3 array (x, y, confidence) and convert to list of lists
    keypoints = keypoints_with_scores[0, 0, :, :].tolist()
    return keypoints

def draw_keypoints(image, keypoints):
    h, w, _ = image.shape
    for x, y in keypoints:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
    return image

import cv2
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub

# Ensure load_movenet_model and preprocess_image are defined elsewhere
# Assume get_keypoints_from_image has been updated as in the previous response

def extract_poses_from_video(video_path, movenet_model, output_json_path):
    """
    Extracts pose keypoints (x, y, confidence) from each frame of a video
    using MoveNet and saves to a JSON file.
    """
    print(f"Extracting poses from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file for pose extraction: {video_path}")
        return None

    all_frames_keypoints = []
    frame_count = 0

    # Load the MoveNet model once
    movenet_model = load_movenet_model() # Assuming this is loaded outside or passed in

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get keypoints using the updated function (returns list of [x, y, conf])
        keypoints_with_confidence = get_keypoints_from_image(frame, movenet_model)

        all_frames_keypoints.append(keypoints_with_confidence)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames for pose extraction.")

    cap.release()
    print(f"Pose extraction complete. Processed {frame_count} frames.")

    try:
        with open(output_json_path, 'w') as f:
            json.dump(all_frames_keypoints, f, indent=2) # Add indent for readability
        print(f"Saved pose data to {output_json_path}")
        return output_json_path
    except IOError:
        print(f"Error: Could not write pose data to {output_json_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while saving pose data: {e}")
        return None
    
    import ffmpeg # For final audio merge
import os

# --- Constants for MoveNet Keypoints (17 points) ---
# These indices are based on typical MoveNet single-pose models.
NOSE_IDX = 0
L_EYE_IDX = 1
R_EYE_IDX = 2
L_EAR_IDX = 3
R_EAR_IDX = 4
L_SHOULDER_IDX = 5
R_SHOULDER_IDX = 6
L_ELBOW_IDX = 7
R_ELBOW_IDX = 8
L_WRIST_IDX = 9
R_WRIST_IDX = 10
L_HIP_IDX = 11
R_HIP_IDX = 12
L_KNEE_IDX = 13
R_KNEE_IDX = 14
L_ANKLE_IDX = 15
R_ANKLE_IDX = 16
MOVENET_KEYPOINTS_COUNT = 17

# Keypoints to be excluded from the distance sum for scoring,
# as they (or their derivatives like midpoints) are used for the transformation.
EXCLUDED_INDICES_FOR_SCORE = {
    L_SHOULDER_IDX, R_SHOULDER_IDX, L_HIP_IDX, R_HIP_IDX
}

# Skeleton edges for drawing the pose
EDGES = [
    (NOSE_IDX, L_EYE_IDX), (L_EYE_IDX, L_EAR_IDX),
    (NOSE_IDX, R_EYE_IDX), (R_EYE_IDX, R_EAR_IDX),
    (NOSE_IDX, L_SHOULDER_IDX), (NOSE_IDX, R_SHOULDER_IDX), # Connect nose to shoulders
    (L_SHOULDER_IDX, R_SHOULDER_IDX),
    (L_SHOULDER_IDX, L_ELBOW_IDX),
    (L_ELBOW_IDX, L_WRIST_IDX),
    (R_SHOULDER_IDX, R_ELBOW_IDX),
    (R_ELBOW_IDX, R_WRIST_IDX),
    (L_SHOULDER_IDX, L_HIP_IDX),
    (R_SHOULDER_IDX, R_HIP_IDX),
    (L_HIP_IDX, R_HIP_IDX),
    (L_HIP_IDX, L_KNEE_IDX),
    (L_KNEE_IDX, L_ANKLE_IDX),
    (R_HIP_IDX, R_KNEE_IDX),
    (R_KNEE_IDX, R_ANKLE_IDX),
]

# --- Helper Functions ---

def load_pose_data_from_json(filepath):
    """Loads pose data from a JSON file.
    Expected format: A list of frames. Each frame is a list of 17 keypoints.
    Each keypoint is a list or tuple [x, y, confidence].
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Basic validation
        if not isinstance(data, list) or (data and not isinstance(data[0], list)):
            print(f"Error: Pose data in {filepath} is not in the expected list-of-lists format.")
            return None
        if data and data[0] and len(data[0]) != MOVENET_KEYPOINTS_COUNT:
            print(f"Error: Frame 0 in {filepath} does not have {MOVENET_KEYPOINTS_COUNT} keypoints.")
            return None
        return data
    except FileNotFoundError:
        print(f"Error: Pose file {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def get_midpoint(p1, p2):
    """Calculates the midpoint between two 2D points."""
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

def draw_pose_with_skeleton(frame, keypoints, point_color=(0, 255, 0), line_color=None, point_radius=6, line_thickness=3, show_indices=False):
    """Draws keypoints and skeleton lines on a frame."""
    if line_color is None:
        line_color = point_color

    h, w, _ = frame.shape

    # Ensure keypoints is a NumPy array for easier slicing
    kps_np = np.array(keypoints)

    drawn_points = {} # Store screen coordinates of valid, drawn points

    for i in range(len(kps_np)):
        # Ensure keypoint has at least x, y, conf
        if len(kps_np[i]) < 3:
            # print(f"Warning: Keypoint {i} has insufficient data: {kps_np[i]}")
            continue # Skip malformed keypoint

        y_norm, x_norm, conf = kps_np[i, 0], kps_np[i, 1], kps_np[i, 2]

        x = int(x_norm * w)
        y = int(y_norm * h)

        # Draw only if confidence is above a threshold and coordinates are valid
        if conf > 0.05 and x >= 0 and y >= 0 and x < frame.shape[1] and y < frame.shape[0]:
            cv2.circle(frame, (x, y), point_radius, point_color, -1)
            if show_indices:
                 cv2.putText(frame, str(i), (x + point_radius, y + point_radius), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            drawn_points[i] = (x,y)

    for kp_start_idx, kp_end_idx in EDGES:
        if kp_start_idx in drawn_points and kp_end_idx in drawn_points:
            start_point = drawn_points[kp_start_idx]
            end_point = drawn_points[kp_end_idx]
            cv2.line(frame, start_point, end_point, line_color, line_thickness)
    return frame



# --- Mock Data Generation and Main Execution Example ---
def generate_mock_pose_data(num_frames, img_width, img_height, movement_scale=50):
    """Generates random-ish mock pose data for testing."""
    poses = []
    # Initial base position for the whole body
    body_center_x = img_width / 2
    body_center_y = img_height / 2

    for frame_num in range(num_frames):
        frame_kps = []
        # Slight overall body sway per frame
        frame_offset_x = np.sin(frame_num * 0.1) * movement_scale * 0.5
        frame_offset_y = np.cos(frame_num * 0.05) * movement_scale * 0.3

        current_body_center_x = body_center_x + frame_offset_x
        current_body_center_y = body_center_y + frame_offset_y

        for i in range(MOVENET_KEYPOINTS_COUNT):
            # Base offset for each keypoint relative to body center (very rough)
            # These are arbitrary, just to make it look somewhat like a pose
            kp_base_offset_x = (i % 5 - 2) * movement_scale * 0.3 # Spread keypoints horizontally
            kp_base_offset_y = (i // 5 - 2) * movement_scale * 0.5 # Spread keypoints vertically

            # Add individual small random jitter for each keypoint
            x = current_body_center_x + kp_base_offset_x + np.random.uniform(-10, 10)
            y = current_body_center_y + kp_base_offset_y + np.random.uniform(-10, 10)

            # Ensure within bounds
            x = np.clip(x, 0, img_width - 1)
            y = np.clip(y, 0, img_height - 1)
            conf = np.random.uniform(0.6, 1.0) # Random confidence
            frame_kps.append([x, y, conf])
        poses.append(frame_kps)
    return poses

def save_pose_data_to_json(filepath, data):
    """Saves pose data to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2) # Add indent for readability
    except IOError:
        print(f"Error: Could not write pose data to {filepath}.")

def create_dummy_video(path, num_d_frames, fps_val, width_val, height_val, color_bgr, text_overlay=""):
    """Creates a simple dummy video file with OpenCV."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps_val, (width_val, height_val))
    if not writer.isOpened():
        print(f"Error: Could not open video writer for {path}")
        return
    for i in range(num_d_frames):
        frame = np.full((height_val, width_val, 3), color_bgr, dtype=np.uint8)
        if text_overlay:
            cv2.putText(frame, f"{text_overlay} Frame: {i+1}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

"""
if __name__ == "__main__":
    print("Setting up dummy files for demonstration...")

    # Dummy video parameters
    dummy_width, dummy_height = 640, 480  # Smaller resolution for faster testing
    dummy_fps = 30
    dummy_num_frames = 90 # 3 seconds of video

    # Define file paths for dummy files
    user_vid_path = "song_user.mp4"
    ref_vid_path = "song_reference.mp4"
    user_pose_json_path = "song_user_poses.json"
    ref_pose_json_path = "song_reference_poses.json"
    final_output_path = "output_dance_comparison.mp4"

    # Create dummy videos
    create_dummy_video(user_vid_path, dummy_num_frames, dummy_fps, dummy_width, dummy_height, (100, 50, 50), "User Video") # Dark blue-ish
    create_dummy_video(ref_vid_path, dummy_num_frames, dummy_fps, dummy_width, dummy_height, (50, 100, 50), "Ref Video") # Dark green-ish
    print(f"Created dummy videos: {user_vid_path}, {ref_vid_path}")

    # Generate and save mock pose data
    # Make reference poses slightly different from user poses for a more interesting test
    user_pose_data = generate_mock_pose_data(dummy_num_frames, dummy_width, dummy_height, movement_scale=40)
    ref_pose_data = generate_mock_pose_data(dummy_num_frames, dummy_width, dummy_height, movement_scale=50)

    save_pose_data_to_json(user_pose_json_path, user_pose_data)
    save_pose_data_to_json(ref_pose_json_path, ref_pose_data)
    print(f"Created dummy pose JSON files: {user_pose_json_path}, {ref_pose_json_path}")

    print("\nStarting main processing pipeline with dummy files...")
    process_videos(
        user_video_path=user_vid_path,
        ref_video_path=ref_vid_path,
        user_pose_path=user_pose_json_path,
        ref_pose_path=ref_pose_json_path,
        output_video_path=final_output_path
    )
    print("\n--- Script Execution Finished ---")
    print(f"Please check for the output file: {final_output_path}")
    print(f"And temporary file (if audio merge failed): temp_video_no_audio.mp4")
    """

import ffmpeg
import os

def create_offset_video(input_video_path, output_video_path, start_offset_seconds):
    """
    Creates a new video file starting from a specified timestamp offset of the input video.

    Args:
        input_video_path (str): The path to the original input video file.
        output_video_path (str): The path where the new offset video will be saved.
        start_offset_seconds (float): The start time offset in seconds.
    """
    print(f"Creating offset video for {input_video_path} starting at {start_offset_seconds} seconds...")
    try:
        # Input stream with start time offset
        input_stream = ffmpeg.input(input_video_path, ss=start_offset_seconds)

        # Output stream - copy video and audio codecs
        output_stream = ffmpeg.output(input_stream, output_video_path, vcodec='copy', acodec='copy')

        # Run the FFmpeg command
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

        print(f"Offset video saved to {output_video_path}")
        return output_video_path

    except ffmpeg.Error as e:
        print("ffmpeg error occurred while creating offset video:")
        if e.stderr:
            print(e.stderr.decode('utf8'))
        else:
            print(str(e))
        return None
    except Exception as e:
        print(f"An unexpected error occurred while creating offset video: {e}")
        return None
    
    import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import ffmpeg # For final audio merge
import os

# --- Constants, Helper Functions (get_video_properties, resize_frame, etc.) ---
# (Assume all previous helper functions and constants like EDGES are defined here)

# Assume load_movenet_model is defined and returns the loaded MoveNet model signature


def process_videos(user_video_path, ref_video_path, output_video_path):
    """
    Processes user and reference videos, extracts poses, performs comparison,
    and creates a comparison video with uniform scaling and frame rate synchronization.
    """
    print("Starting video processing and pose extraction...")

    # 1. Load MoveNet Model
    movenet_model = load_movenet_model()
    if movenet_model is None:
        print("Error: Could not load MoveNet model. Aborting.")
        return

    # 2. Define Paths for Pose Data JSON files
    user_pose_json_path = "user_poses.json"
    ref_pose_json_path = "ref_poses.json"

    # 3. Extract Poses from Videos and Save to JSON
    # **Important:** For accurate synchronization later, it's better to extract
    # poses from the ORIGINAL videos at their full frame rates first.
    # The pose data will contain the poses for every frame captured by MoveNet.
    user_pose_file = extract_poses_from_video(user_video_path, movenet_model, user_pose_json_path)
    ref_pose_file = extract_poses_from_video(ref_video_path, movenet_model, ref_pose_json_path)

    if user_pose_file is None or ref_pose_file is None:
        print("Error during pose extraction. Aborting video processing.")
        return

    # 4. Load Pose Data from JSON files
    user_poses_all_frames = load_pose_data_from_json(user_pose_file)
    ref_poses_all_frames = load_pose_data_from_json(ref_pose_file)

    if user_poses_all_frames is None or ref_poses_all_frames is None:
        print("Error loading pose data from JSON. Aborting.")
        if os.path.exists(user_pose_json_path): os.remove(user_pose_json_path)
        if os.path.exists(ref_pose_json_path): os.remove(ref_pose_json_path)
        return

    # 5. Open Video Files for Processing
    cap_user = cv2.VideoCapture(user_video_path)
    cap_ref = cv2.VideoCapture(ref_video_path)

    if not cap_user.isOpened():
        print(f"Error: Could not open user video file: {user_video_path}")
        if os.path.exists(user_pose_json_path): os.remove(user_pose_json_path)
        if os.path.exists(ref_pose_json_path): os.remove(ref_pose_json_path)
        return
    if not cap_ref.isOpened():
        print(f"Error: Could not open reference video file: {ref_video_path}")
        if cap_user.isOpened(): cap_user.release()
        if os.path.exists(user_pose_json_path): os.remove(user_pose_json_path)
        if os.path.exists(ref_pose_json_path): os.remove(ref_pose_json_path)
        return

    # 6. Get Video Properties and Determine Target FPS and Dimensions
    user_width, user_height, user_fps = get_video_properties(user_video_path)
    ref_width, ref_height, ref_fps = get_video_properties(ref_video_path)

    if user_width is None or ref_width is None:
        print("Error getting video properties. Aborting.")
        cap_user.release()
        cap_ref.release()
        if os.path.exists(user_pose_json_path): os.remove(user_pose_json_path)
        if os.path.exists(ref_pose_json_path): os.remove(ref_pose_json_path)
        return

    # Use reference video's dimensions as the target for display panels
    target_width, target_height = ref_width, ref_height
    # Use the maximum FPS of the two videos as the output FPS for smoother playback
    output_fps = max(user_fps, ref_fps)

    print(f"User video FPS: {user_fps}, Reference video FPS: {ref_fps}, Output FPS: {output_fps}")
    print(f"Target display dimensions: {target_width}x{target_height}")


    # 7. Setup Output Video (Two panels)
    temp_output_no_audio_path = "temp_video_no_audio.mp4"
    out_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # The output video will have a width of target_width * 2
    video_writer = cv2.VideoWriter(temp_output_no_audio_path, out_fourcc, output_fps, (target_width * 2, target_height))

    # 8. Process Frames and Pose Data
    total_score = 0.0
    processed_frame_count = 0 # Count of frames written to the output video

    # Synchronize frames based on timestamps
    user_frame_index = 0
    ref_frame_index = 0

    # Calculate the duration of each frame
    user_frame_duration = 1.0 / user_fps if user_fps > 0 else float('inf')
    ref_frame_duration = 1.0 / ref_fps if ref_fps > 0 else float('inf')

    # Track time in both videos
    user_time = 0.0
    ref_time = 0.0
    output_time = 0.0
    output_frame_duration = 1.0 / output_fps if output_fps > 0 else float('inf')

    print("Starting frame processing and synchronization...")

    # Read the first frames
    ret_user, frame_user = cap_user.read()
    ret_ref, frame_ref = cap_ref.read()

    if ret_ref:
      frame_ref = cv2.flip(frame_ref, 1) # 1 flips this frame about the y-axis

    while ret_user and ret_ref and user_frame_index < len(user_poses_all_frames) and ref_frame_index < len(ref_poses_all_frames):

        # Get current pose data based on current frame indices
        current_user_pose_list = user_poses_all_frames[user_frame_index]
        current_ref_pose_list = ref_poses_all_frames[ref_frame_index]

        # Validate pose data for the current frames
        if not isinstance(current_user_pose_list, list) or len(current_user_pose_list) != MOVENET_KEYPOINTS_COUNT or \
           not isinstance(current_ref_pose_list, list) or len(current_ref_pose_list) != MOVENET_KEYPOINTS_COUNT:
            print(f"Warning: Invalid pose data format for user frame {user_frame_index} or ref frame {ref_frame_index}. Skipping to next pose data.")
             # Advance pose data indices even if frame is skipped
            user_frame_index += 1
            ref_frame_index += 1
            # Also advance video capture if needed to keep pace with time
            while user_time < output_time and ret_user:
                ret_user, frame_user = cap_user.read()
                user_time += user_frame_duration
            while ref_time < output_time and ret_ref:
                ret_ref, frame_ref = cap_ref.read()
                ref_time += ref_frame_duration
            output_time += output_frame_duration
            continue # Skip processing this frame pair

        current_user_pose = np.array(current_user_pose_list, dtype=np.float32)
        current_ref_pose = np.array(current_ref_pose_list, dtype=np.float32)

        for i in range(current_ref_pose.shape[0]): # flip x-coordinates of all reference points
             current_ref_pose[i, 1] = 1.0 - current_ref_pose[i, 1]

        # Ensure keypoints have x,y,conf structure
        if current_user_pose.shape[1] < 3 or current_ref_pose.shape[1] < 3:
            print(f"Warning: Keypoints in user frame {user_frame_index} or ref frame {ref_frame_index} do not have (x,y,conf). Skipping.")
             # Advance pose data indices and video capture
            user_frame_index += 1
            ref_frame_index += 1
            while user_time < output_time and ret_user:
                ret_user, frame_user = cap_user.read()
                user_time += user_frame_duration
            while ref_time < output_time and ret_ref:
                ret_ref, frame_ref = cap_ref.read()
                ref_time += ref_frame_duration
            output_time += output_frame_duration
            continue # Skip processing this frame pair


        # 9. Transformation using Affine Transform (for SCORING and OVERLAY)
        # ... (affine transformation calculation using your chosen points, e.g., Shoulders and Mid-Hip)
        user_l_shoulder = current_user_pose[L_SHOULDER_IDX, :2]
        user_r_shoulder = current_user_pose[R_SHOULDER_IDX, :2]
        user_l_hip, user_r_hip = current_user_pose[L_HIP_IDX, :2], current_user_pose[R_HIP_IDX, :2]
        user_mid_hip = get_midpoint(user_l_hip, user_r_hip)

        ref_l_shoulder = current_ref_pose[L_SHOULDER_IDX, :2]
        ref_r_shoulder = current_ref_pose[R_SHOULDER_IDX, :2]
        ref_l_hip, ref_r_hip = current_ref_pose[L_HIP_IDX, :2], current_ref_pose[R_HIP_IDX, :2]
        ref_mid_hip = get_midpoint(ref_l_hip, ref_r_hip)

        pts_user = np.float32([user_l_shoulder, user_r_shoulder, user_mid_hip]) # Use your preferred points
        pts_ref = np.float32([ref_l_shoulder, ref_r_shoulder, ref_mid_hip])   # Use your preferred points

        # Check for collinearity (adapt to your chosen points)
        user_area = 0.5 * abs(pts_user[0,0]*(pts_user[1,1] - pts_user[2,1]) + \
                                pts_user[1,0]*(pts_user[2,1] - pts_user[0,1]) + \
                                pts_user[2,0]*(pts_user[0,1] - pts_user[1,1]))

        if user_area < 1e-2:
             M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32) # Identity transform if collinear
             # print(f"Warning: User transformation points are collinear at time {user_time:.2f}s. Using identity transform.")
        else:
             M = cv2.getAffineTransform(pts_user, pts_ref)


        # Apply the transformation to ALL user keypoints for SCORING AND OVERLAY
        user_pose_transformed_np = np.copy(current_user_pose)
        for kp_idx in range(MOVENET_KEYPOINTS_COUNT):
            original_xy = current_user_pose[kp_idx, :2]
            homogeneous_coord = np.array([original_xy[0], original_xy[1], 1.0], dtype=np.float32)
            transformed_xy = (M @ homogeneous_coord)
            user_pose_transformed_np[kp_idx, 0] = transformed_xy[0]
            user_pose_transformed_np[kp_idx, 1] = transformed_xy[1]
            # Confidence remains the same

        # 10. Distance Calculation and Score (Use the TRANSFORMED pose for scoring)
        frame_distance_sum = 0.0
        epsilon = 1e-6

        for i in range(MOVENET_KEYPOINTS_COUNT):
            if i not in EXCLUDED_INDICES_FOR_SCORE: # Ensure EXCLUDED_INDICES_FOR_SCORE is updated for your chosen points
                # Ensure keypoints have at least x,y,conf before accessing confidence
                if user_pose_transformed_np.shape[1] >= 3 and current_ref_pose.shape[1] >= 3:
                     # Consider confidence in scoring? (Optional)
                     # conf_user = user_pose_transformed_np[i, 2]
                     # conf_ref = current_ref_pose[i, 2]
                     # if conf_user > 0.2 and conf_ref > 0.2:
                     p_user_transformed = user_pose_transformed_np[i, :2]
                     p_ref = current_ref_pose[i, :2]
                     dist = np.linalg.norm(p_user_transformed - p_ref)
                     frame_distance_sum += dist
                else:
                    print(f"Warning: Insufficient keypoint data for scoring at time {output_time:.2f}s.")


        if frame_distance_sum > epsilon:
            frame_score_component = 1.0 / frame_distance_sum
        else:
            frame_score_component = 1.0 / epsilon

        total_score += frame_score_component


        # 11. Visualization

        # --- Create the left panel (Reference with both skeletons) ---
        frame_left_panel = frame_ref.copy()
        # Draw Reference pose (e.g., green)
        frame_left_panel = draw_pose_with_skeleton(frame_left_panel, current_ref_pose, point_color=(0, 255, 0), line_color=(0,200,0), show_indices=False)
        # Draw TRANSFORMED User pose (e.g., blue)
        frame_left_panel = draw_pose_with_skeleton(frame_left_panel, user_pose_transformed_np, point_color=(255, 0, 0), line_color=(200,0,0), show_indices=False)
        # ---------------------------------------------------------------------

        # --- Create the right panel (User with original skeleton - scaled) ---
        # Uniformly scale the user frame to fit within the target dimensions while maintaining aspect ratio
        user_aspect_ratio = user_width / user_height
        target_aspect_ratio = target_width / target_height

        if user_aspect_ratio > target_aspect_ratio:
            # User video is wider than target, scale by width
            scale_factor = target_width / user_width
            scaled_width = target_width
            scaled_height = int(user_height * scale_factor)
        else:
            # User video is taller than target or same aspect ratio, scale by height
            scale_factor = target_height / user_height
            scaled_height = target_height
            scaled_width = int(user_width * scale_factor)

        # Resize the user frame
        frame_user_scaled = cv2.resize(frame_user, (scaled_width, scaled_height))

        # Create a blank canvas for the right panel (target dimensions)
        frame_right_panel = np.full((target_height, target_width, 3), (0, 0, 0), dtype=np.uint8) # Black background

        # Calculate padding to center the scaled user frame
        pad_x = (target_width - scaled_width) // 2
        pad_y = (target_height - scaled_height) // 2

        # Place the scaled user frame onto the right panel canvas
        frame_right_panel[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = frame_user_scaled

        # Draw the ORIGINAL user pose on the scaled user frame within the right panel canvas
        # Note: The keypoint coordinates in current_user_pose are normalized to the ORIGINAL user frame size.
        # draw_pose_with_skeleton needs the frame it's drawing on's dimensions for scaling.
        # So, we can draw directly on the frame_user_scaled, but need to adjust keypoint coords if drawing on frame_right_panel.
        # Let's draw directly on the frame_user_scaled, then put it on the right panel. Or better, adjust keypoint coords.
        # A simpler approach is to draw directly on the frame_right_panel, but scale the original user keypoints
        # by the dimensions of the target_width x target_height panel, and then apply the offset (pad_x, pad_y).

        # Let's modify draw_pose_with_skeleton slightly to handle an offset and target size if needed
        # For now, let's stick to the current draw_pose_with_skeleton which works on the frame it's given.

        # We need to draw the original user pose on the *scaled* user frame within the padded right panel.
        # The simplest way is to:
        # 1. Create the full size right panel (black background).
        # 2. Place the scaled user frame onto it.
        # 3. Draw the original user pose, but scaled to the *original* user video dimensions and then offset.

        # Let's redraw the right panel to correctly place the scaled user video and draw the pose.
        frame_right_panel = np.full((target_height, target_width, 3), (0, 0, 0), dtype=np.uint8) # Black background
        frame_right_panel[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = frame_user_scaled

        # Draw the original user pose, scaled and offset for the right panel
        # We need to scale the original user keypoints (0-1 range) by the *original* user dimensions,
        # then scale that result by the factor applied to the frame, and add the padding.
        # Or, scale the original user keypoints by the *target* dimensions and add padding.
        # The draw_pose_with_skeleton expects keypoints normalized to the frame it's drawing on.

        # Let's draw the original user pose on the scaled frame *before* padding,
        # then place the annotated scaled frame on the padded panel.

        frame_user_scaled_annotated = draw_pose_with_skeleton(frame_user_scaled.copy(), current_user_pose, point_color=(255, 0, 0), line_color=(200,0,0), show_indices=False)
        # Now place this annotated scaled frame onto the right panel
        frame_right_panel[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = frame_user_scaled_annotated
        # -------------------------------------------------------------


        # Create a two-panel side-by-side image
        combined_frame = np.zeros((target_height, target_width * 2, 3), dtype=np.uint8) # Two panels wide
        combined_frame[:, :target_width] = frame_left_panel  # Place the modified left panel
        combined_frame[:, target_width:] = frame_right_panel # Place the right panel


        # Add score text (centered across two panels)
        score_text = f"Score: {total_score:.2f}"

        # Calculate font scale based on video height
        font_scale = 0.002 * target_height # Example: A simple linear scaling
        if font_scale < 0.5: font_scale = 0.5 # Minimum font scale

        # Calculate thickness based on font scale
        thickness = max(1, int(font_scale * 1.5)) # Example: Thickness scales with font size

        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (target_width * 2 - text_size[0]) // 2 # Centered across two panels
        text_y = int(target_height * 0.1) # Keep it positioned near the top

        cv2.putText(combined_frame, score_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
        cv2.putText(combined_frame, score_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,0), thickness, cv2.LINE_AA)


        # Add panel labels (optional)
        cv2.putText(combined_frame, "Reference & User Overlay", (target_width // 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, "User", (target_width + target_width // 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Add frame counter (adjust position)
        frame_info_text = f"Frame: {processed_frame_count + 1}" # Display count of processed output frames
        cv2.putText(combined_frame, frame_info_text, (10, target_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)


        video_writer.write(combined_frame)
        processed_frame_count += 1 # Increment the count of output frames

        # --- Advance video capture based on time ---
        output_time += output_frame_duration

        # Read frames until user_time and ref_time catch up to output_time
        while user_time < output_time and ret_user:
            ret_user, frame_user = cap_user.read()
            if ret_user:
                user_time += user_frame_duration
                user_frame_index += 1 # Advance user pose index with frame read

        while ref_time < output_time and ret_ref:
            ret_ref, frame_ref = cap_ref.read()
            if ret_ref:
                frame_ref = cv2.flip(frame_ref, 1) # flip reference frame about the y axis
                ref_time += ref_frame_duration
                ref_frame_index += 1 # Advance ref pose index with frame read

        # If either video ran out of frames, the loop condition will become false
        # If pose data ran out, the loop condition will become false


    # 12. Release resources
    print("Releasing video captures and writer...")
    cap_user.release()
    cap_ref.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # 13. Add audio using ffmpeg-python (Keep this logic)
    # ... (audio merge logic)
    print(f"Video processing complete. Final score: {total_score:.2f}")
    print("Adding audio to video...")
    try:
        if not os.path.exists(temp_output_no_audio_path):
            print(f"Error: Temporary video file {temp_output_no_audio_path} not found. Cannot add audio.")
            return

        input_video_stream = ffmpeg.input(temp_output_no_audio_path)
        # Use audio from reference video by default
        audio_source_video = ref_video_path if os.path.exists(ref_video_path) else user_video_path

        if not os.path.exists(audio_source_video):
            print(f"Error: Audio source video {audio_source_video} not found. Outputting video without audio.")
            os.rename(temp_output_no_audio_path, output_video_path) # Rename temp to final
            print(f"Final video (no audio) saved to {output_video_path}")
            return

        input_audio_stream = ffmpeg.input(audio_source_video).audio

        stream = ffmpeg.output(input_video_stream, input_audio_stream, output_video_path, vcodec='copy', acodec='aac', strict='experimental')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        print(f"Final video with audio saved to {output_video_path}")
        if os.path.exists(temp_output_no_audio_path): # Clean up temp file
            os.remove(temp_output_no_audio_path)

    except ffmpeg.Error as e:
        print("ffmpeg error occurred while adding audio:")
        if e.stderr:
            print(e.stderr.decode('utf8'))
        else:
            print(str(e))
        print(f"Video without audio was saved to {temp_output_no_audio_path}")
        print(f"You might need to rename it to {output_video_path} manually if you want to keep it.")
    except Exception as e:
        print(f"An unexpected error occurred during audio merge: {e}")
        print(f"Video without audio was saved to {temp_output_no_audio_path}")

    finally:
         # Clean up generated pose files regardless of audio merge success
        if os.path.exists(user_pose_json_path): os.remove(user_pose_json_path)
        if os.path.exists(ref_pose_json_path): os.remove(ref_pose_json_path)


# --- Main Execution Example (Modified to use actual video paths) ---
if __name__ == "__main__":
    print("Starting dance comparison pipeline...")

    # Define file paths for your actual videos
    # Replace these with the actual paths to your uploaded videos
    user_video_path = "/content/drive/MyDrive/CalPoly/YEAR4/justdance_videos/SoWhat_user.mp4"
    ref_video_path = "/content/drive/MyDrive/CalPoly/YEAR4/justdance_videos/SoWhat_ref_aligned.mp4"
    final_output_path = "/content/drive/MyDrive/CalPoly/YEAR4/justdance_videos/SoWhat_out_FINAL.mp4"

    # Check if input files exist
    if not os.path.exists(user_video_path):
        print(f"Error: User video not found at {user_video_path}")
    elif not os.path.exists(ref_video_path):
        print(f"Error: Reference video not found at {ref_video_path}")
    else:
        print("\nStarting main processing pipeline with provided video files...")
        process_videos(user_video_path=user_video_path, ref_video_path=ref_video_path, output_video_path=final_output_path)
        print("\n--- Script Execution Finished ---")
        print(f"Please check for the output file: {final_output_path}")


import json

user_json_path = "user_poses.json"
ref_json_path = "ref_poses.json"

try:
    with open(user_json_path, 'r') as f:
        user_poses_data = json.load(f)
    print(f"Loaded user pose data. Number of frames: {len(user_poses_data)}")
    # Print data for the first few frames to inspect
    print("First 5 user frames data:")
    for i, frame_data in enumerate(user_poses_data[:5]):
        print(f"  Frame {i}: {frame_data}")

except FileNotFoundError:
    print(f"Error: {user_json_path} not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {user_json_path}.")
except Exception as e:
    print(f"An error occurred: {e}")

print("-" * 20) # Separator

try:
    with open(ref_json_path, 'r') as f:
        ref_poses_data = json.load(f)
    print(f"Loaded reference pose data. Number of frames: {len(ref_poses_data)}")
    # Print data for the first few frames to inspect
    print("First 5 reference frames data:")
    for i, frame_data in enumerate(ref_poses_data[:5]):
        print(f"  Frame {i}: {frame_data}")

except FileNotFoundError:
    print(f"Error: {ref_json_path} not found.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {ref_json_path}.")
except Exception as e:
    print(f"An error occurred: {e}")