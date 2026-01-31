from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
import pandas as pd

def calculate_congestion(tracking_speeds: dict) -> float:
    """
    Calculate congestion level based on average vehicle speeds. The idea behind this is that
    the more congested the traffic is, the lower the average speed of vehicles will be.
    
    Args:
        tracking_speeds (dict): Dictionary mapping vehicle labels to list of speeds (Km/h)
    
    Returns:
        float: Normalized congestion level (0.0 = free flow, 1.0 = fully congested)
    """
    if not tracking_speeds:
        return 0.0
    
    average_speed = 0.0
    max_speed = 97  # Maximum expected speed (Km/h) based on this specific road 
    
    # Calculate average speed per vehicle, then overall average
    for v_list in tracking_speeds.values():
        average_speed += (sum(v_list) / len(v_list))
    average_speed = average_speed / len(tracking_speeds)
    
    # Normalize by max speed and return inverse (higher speed = lower congestion)
    return average_speed / max_speed

def get_xy_in_meter(x, y, roi) -> tuple:
    """
    Convert pixel coordinates to real-world coordinates (in meters).
    Uses perspective transform based on road ROI to map pixel coords to world coords.
    
    Args:
        x (int): X pixel coordinate
        y (int): Y pixel coordinate
        roi (np.ndarray): Road region of interest polygon (pixel coordinates)
    
    Returns:
        tuple: (x_meters, y_meters) - Real-world coordinates
    """
    # Define real-world road dimensions
    world_points = np.array([
        [0, 0],           # Top-left
        [23, 0],          # Top-right (23 meters wide)
        [23, 31.5],       # Bottom-right (31.5 meters long)
        [0, 31.5]         # Bottom-left
    ], np.float32)
    
    # Convert ROI polygon to float32 for transformation
    road_points = roi.astype(np.float32)
    
    # Calculate perspective transformation matrix
    mask = cv2.getPerspectiveTransform(road_points, world_points)
    
    # Apply transformation to pixel coordinates
    pixels = np.array([[x, y]], dtype=np.float32)
    pixels = pixels.reshape(-1, 1, 2)
    world_coords = cv2.perspectiveTransform(pixels, mask)
    
    return world_coords[0][0][0], world_coords[0][0][1]

def get_lane(box_center: tuple, lane_polygons: dict) -> str:
    """
    Determine which lane a vehicle belongs to based on its center position.
    
    Args:
        box_center (tuple): (cx, cy) - Center coordinates of detection box
        lane_polygons (dict): Dictionary mapping lane names to polygon coordinates
    
    Returns:
        str: Lane name (e.g., 'lane1', 'lane2', etc.)
    """
    cx, cy = box_center
    for lane, lane_polygon in lane_polygons.items():
        # Check if point is inside polygon (returns >= 0 if inside)
        if cv2.pointPolygonTest(lane_polygon, (cx, cy), False) >= 0:
            return lane

def check_lane_changes(tracking_lanes: dict) -> None:
    """
    Detect and log when vehicles change lanes.
    
    Args:
        tracking_lanes (dict): Dictionary mapping vehicle labels to list of lane history
    """
    for label, lanes in tracking_lanes.items():
        if len(lanes) < 2:
            continue
        # Check if the most recent lane differs from the previous one
        if lanes[-1] != lanes[-2]:
            print(f"{label} changed from {lanes[-2]} to {lanes[-1]}")

def define_lane_polygons() -> dict:
    """Define the 5 lanes of the road as polygon vertices (in pixel coordinates)"""
    return {
        "lane1": np.array([[131, 101], [133, 132], [134, 166], [145, 208],
                           [195, 211], [173, 170], [164, 130], [166, 98]], dtype=np.int32),
        "lane2": np.array([[166, 98], [164, 130], [173, 170], [195, 211],
                           [252, 211], [213, 168], [192, 128], [185, 97]], dtype=np.int32),
        "lane3": np.array([[185, 97], [192, 128], [213, 168], [252, 211],
                           [285, 208], [243, 165], [216, 128], [206, 95]], dtype=np.int32),
        "lane4": np.array([[206, 95], [216, 128], [243, 165], [285, 208],
                           [315, 199], [271, 163], [241, 126], [225, 95]], dtype=np.int32),
        "lane5": np.array([[225, 95], [241, 126], [271, 163], [315, 199],
                           [318, 172], [296, 136], [266, 100], [247, 97]], dtype=np.int32)
    }

def define_road_roi() -> np.ndarray:
    """Define the main road region of interest (ROI) for analysis"""
    return np.array([
        [133, 101],   # Top-left
        [249, 97],    # Top-right
        [320, 175],   # Bottom-right
        [130, 210]    # Bottom-left
    ], dtype=np.int32)

# MAIN PIPELINE

if __name__ == "__main__":
    # Configuration parameters
    VIDEOS_DIR = "video"
    MODEL_PATH = "yolov10s.pt"
    CONF_THRESHOLD = 0.4
    SPEED_CORRECTION_FACTOR = 130 # Calibration factor for speed based on maximum speed observed and maximum expected speed
    VISUALIZATION_SCALE = 3  # Scale for display (3x larger than original)
    VEHICLE_CLASSES = [2, 3, 5, 7]  # YOLO class IDs: car, motorcycle, bus, truck
    
    # Initialize output report DataFrame
    output_report = pd.DataFrame(
        columns=["Video Name", "Vehicle Count", "Traffic Density (Vehicle / unit road Km)", 
                 "Congestion Level (%)"]
    )
    
    # Get all video files
    files = [f for f in os.listdir(VIDEOS_DIR) 
             if os.path.isfile(os.path.join(VIDEOS_DIR, f))]
    
    # Define road geometry
    lane_polygons = define_lane_polygons()
    road_roi = define_road_roi()
    
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
    # PROCESS EACH VIDEO FROM DATASET `video` DIRECTORY
    for video_idx, video_name in enumerate(files, 1):
        VIDEO_PATH = os.path.join(VIDEOS_DIR, video_name)
        
        # Open video
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Skip first frame (often corrupted) as mentioned in the Kaggle project description
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        
        # Get video properties
        num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Resolution: {width} x {height}, FPS: {fps}, Frames: {num_of_frames}")
        
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video {VIDEO_PATH}")
            continue
        
        # Initialize tracking data structures
        vehicles = set()                    # Unique vehicle IDs in this video
        tracking_centers = {}               # Vehicle ID -> list of (x, y) pixel positions for the center of each box
        tracking_speeds = {}                # Vehicle ID -> list of speeds (Km/h)
        tracking_lanes = {}                 # Vehicle ID -> list of lane assignments
        
        # Initialize video writer for output
        out = None
        
        # FRAME-BY-FRAME PROCESSING
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO DETECTION AND TRACKING 
            results = model.track(
                source=frame,
                persist=True,
                conf=CONF_THRESHOLD,
                verbose=False,
                tracker="botsort.yaml"
            )
            detections = results[0].boxes
            
            # Initialize lane occupancy counter for this frame
            lane_occupancy = {f'lane{i}': 0 for i in range(1, 6)}
            
            
            # PREPARE VISUALIZATION 
            frame_h, frame_w = frame.shape[:2]
            
            # Scale frame for display
            frame_display = cv2.resize(frame, 
                                      (int(frame_w * VISUALIZATION_SCALE), 
                                       int(frame_h * VISUALIZATION_SCALE)),
                                      interpolation=cv2.INTER_LINEAR)
            
            # Draw road ROI boundary
            cv2.polylines(frame_display, [road_roi * VISUALIZATION_SCALE], 
                         isClosed=True, color=(0, 0, 255), thickness=1)
            
            # Create lane overlay with transparency
            overlay = frame_display.copy()
            colors = [(255, 0, 255), (0, 255, 255), (255, 128, 0), 
                     (128, 0, 255), (0, 255, 128)]
            for (lane, lane_polygon), color in zip(lane_polygons.items(), colors):
                scaled_lane_polygon = (lane_polygon * VISUALIZATION_SCALE).astype(np.int32)
                cv2.fillPoly(overlay, [scaled_lane_polygon], color)
            
            # Apply transparency to lane overlay
            # These will display a transparent colored overlay on each road lane
            alpha = 0.15
            main_road_mask = np.zeros((int(frame_h * VISUALIZATION_SCALE), 
                                      int(frame_w * VISUALIZATION_SCALE)), 
                                      dtype=np.uint8)
            cv2.fillPoly(main_road_mask, [road_roi * VISUALIZATION_SCALE], 255)
            overlay_roi = cv2.bitwise_and(overlay, overlay, mask=main_road_mask)
            cv2.addWeighted(overlay_roi, alpha, frame_display, 1 - alpha, 0, frame_display)
            
            # Initialize video writer on first frame
            if out is None:
                output_dir = "output_videos"
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
                output_path = os.path.join(output_dir, f"{base_name}_tracked.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 10,
                                     (frame_display.shape[1], frame_display.shape[0]))
            
            
            # PROCESS DETECTIONS
            if detections is not None and detections.id is not None:
                for box, track_id in zip(detections.xyxy, detections.id):
                    cls_id = int(detections.cls[list(detections.id).index(track_id)])
                    track_id = int(track_id)
                    
                    # Only process vehicle classes
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)  # Center X (pixels)
                    cy = int((y1 + y2) / 2)  # Center Y (pixels)
                    
                    # Check if vehicle is within road ROI
                    if cv2.pointPolygonTest(road_roi, (cx, cy), False) < 0:
                        continue
                    
                    # Add vehicle to tracking
                    vehicles.add(track_id)
                    label_id = f"{model.names[cls_id].title()} #{track_id}"
                    
                    # Convert pixel coordinates to real-world meters
                    new_center = get_xy_in_meter(cx, cy, roi=road_roi)
                    
                    # Determine lane and update occupancy
                    lane = get_lane((cx, cy), lane_polygons=lane_polygons)
                    if lane:
                        lane_occupancy[lane] += 1
                        
                        # Track lane assignments
                        if label_id in tracking_lanes:
                            tracking_lanes[label_id].append(lane)
                        else:
                            tracking_lanes[label_id] = [lane]
                    
                    # Calculate speed
                    speed = 0.0
                    if label_id in tracking_centers:
                        # Get previous position and convert to meters
                        old_center = tracking_centers[label_id][-1]
                        old_center_meters = get_xy_in_meter(old_center[0], old_center[1], roi=road_roi)
                        
                        # Calculate Euclidean distance
                        distance = math.sqrt((old_center_meters[0] - new_center[0])**2 + 
                                           (old_center_meters[1] - new_center[1])**2)
                        
                        # Convert to Km/h: distance_m * factor * 3.6 / fps
                        speed = distance * SPEED_CORRECTION_FACTOR * 3.6 / fps
                        tracking_speeds[label_id].append(speed)
                    else:
                        tracking_speeds[label_id] = [0.0]
                        tracking_centers[label_id] = []
                    
                    # Store current position
                    tracking_centers[label_id].append((cx, cy))
                    

                    # DRAW VEHICLE ANNOTATIONS 
                    # Draw trajectory (connected line of past positions)
                    if len(tracking_centers[label_id]) > 1:
                        for i in range(1, len(tracking_centers[label_id])):
                            pt1 = tracking_centers[label_id][i - 1]
                            pt2 = tracking_centers[label_id][i]
                            cv2.line(frame_display,
                                   (int(pt1[0] * VISUALIZATION_SCALE), 
                                    int(pt1[1] * VISUALIZATION_SCALE)),
                                   (int(pt2[0] * VISUALIZATION_SCALE), 
                                    int(pt2[1] * VISUALIZATION_SCALE)),
                                   (0, 0, 255), thickness=2)
                    
                    # Draw bounding box
                    cv2.rectangle(frame_display, 
                                (int(x1 * VISUALIZATION_SCALE), int(y1 * VISUALIZATION_SCALE)),
                                (int(x2 * VISUALIZATION_SCALE), int(y2 * VISUALIZATION_SCALE)),
                                (255, 0, 0), thickness=2)
                    
                    # Draw label with speed
                    label_text = f"{model.names[cls_id].title()} [{track_id}] - {speed:.2f} Km/h"
                    cv2.putText(frame_display,
                              label_text,
                              (int(x1 * VISUALIZATION_SCALE), 
                               int((y1 - 10) * VISUALIZATION_SCALE)),
                              cv2.FONT_HERSHEY_DUPLEX,
                              0.6, (255, 0, 0), thickness=2)
            
            # Track lane changes
            check_lane_changes(tracking_lanes=tracking_lanes)
            

            # PRESENT STATS IN THE TOP OF THE FRAME
            # Prepare statistics text
            stats_text = (f"Vehicles: {len(vehicles)} || "
                         f"Lane1:{lane_occupancy['lane1']} Lane2:{lane_occupancy['lane2']} "
                         f"Lane3:{lane_occupancy['lane3']} Lane4:{lane_occupancy['lane4']} "
                         f"Lane5:{lane_occupancy['lane5']}")
            
            cv2.putText(frame_display, stats_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX,
                       0.7, (0, 0, 255), thickness=2)
            
            # Display frame
            cv2.imshow(f"Processing Video {video_idx}/{len(files)}: {video_name}", frame_display)
            
            # Write frame to output video
            if out is not None:
                out.write(frame_display)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # CALCULATE AND PRINT STATISTICS
        
        print(f"  Vehicles detected: {len(vehicles)}")
        traffic_density = round(len(vehicles) * 1000 / 31.5)
        print(f"  Traffic Density: {traffic_density} Vehicle/unit road (Km)")
        
        free_flow_level = calculate_congestion(tracking_speeds=tracking_speeds)
        congestion_percent = (1 - free_flow_level) * 100
        print(f"  Congestion level: {congestion_percent:.3f}%\n")
        
        # Add results to output report
        output_report.loc[len(output_report)] = [
            video_name[:-4],  # Remove file extension
            len(vehicles),
            traffic_density,
            round(congestion_percent, 3)
        ]
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    
    
    # SAVE FINAL REPORT
    report_output_dir = "output_reports"
    os.makedirs(report_output_dir, exist_ok=True)
    report_output_path = os.path.join(report_output_dir, "traffic_report.csv")
    output_report.to_csv(report_output_path, index=False)
    print(f"Report saved to: {report_output_path}")
    print("\nProcessing complete!")

    output_report.to_csv(report_output_path, index=False)
    print(f"Report saved to {report_output_path}")