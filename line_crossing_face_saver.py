#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import gi
import pyds
import math
import time
import datetime
import requests
from io import BytesIO
import threading
import configparser
from collections import defaultdict

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# Global constants
PGIE_CLASS_ID_PERSON = 0
FACE_SAVE_DIR = "saved_faces"
LINE_Y_POSITION = 0.5  # Line position as fraction of frame height (adjust as needed)
FACE_SAVE_INTERVAL = 10  # Minimum frames between face saves for same ID
MAX_SAVED_FACES = 5  # Maximum faces to save per tracked ID

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create directory for saving faces
os.makedirs(FACE_SAVE_DIR, exist_ok=True)

# Tracker dictionaries
class_tracker = {}  # For tracking which objects crossed the line
last_saved = {}     # For tracking when we last saved face for this ID
saved_count = {}    # Count of saved faces per ID

def download_yolov8_model():
    """Download YOLOv8 weights and create config files if needed"""
    if not os.path.exists("yolov8n.wts") or not os.path.exists("yolov8n.cfg"):
        print("Downloading YOLOv8n model...")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Download YOLOv8n weights using PyTorch hub
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
            
            # Save model weights in format compatible with DeepStream
            print("Converting model to DeepStream format...")
            model.model.save("yolov8n.wts")
            
            # Create config file for DeepStream
            with open("yolov8n.cfg", "w") as cfg_file:
                cfg_file.write("[net]\n")
                cfg_file.write("width=640\n")
                cfg_file.write("height=640\n")
                cfg_file.write("channels=3\n")
                cfg_file.write("batch=1\n")
                
            print("YOLOv8n model ready for DeepStream")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download YOLOv8n model manually and place in the working directory")
            sys.exit(1)

def create_deepstream_config():
    """Create DeepStream config file"""
    config = configparser.ConfigParser()
    
    # Application settings
    config['application'] = {
        'enable-perf-measurement': '1',
        'perf-measurement-interval-sec': '5'
    }
    
    # Source settings
    config['source0'] = {
        'enable': '1',
        'type': '4',  # URI type (can be changed to 1 for camera)
        'uri': 'file:///path/to/your/video.mp4',  # Replace with your video path
        'num-sources': '1',
        'gpu-id': '0'
    }
    
    # Primary detector settings (YOLOv8)
    config['primary-gie'] = {
        'enable': '1',
        'gpu-id': '0',
        'interval': '0',
        'gie-unique-id': '1',
        'model-engine-file': 'models/yolov8n.engine',
        'labelfile-path': 'labels.txt',
        'config-file': 'yolov8_config.txt',
        'batch-size': '1',
        'network-mode': '0',  # 0 for detection, 1 for detection + classification
        'process-mode': '1',  # 1 for primary detector
        'model-color-format': '0'  # 0 = RGB
    }
    
    # Tracker settings
    config['tracker'] = {
        'enable': '1',
        'tracker-width': '640',
        'tracker-height': '384',
        'gpu-id': '0',
        'll-lib-file': '/opt/nvidia/deepstream/deepstream-6.2/lib/libnvds_nvmultiobjecttracker.so',
        'll-config-file': 'tracker_config.yml',
        'display-tracking-id': '1'
    }
    
    # Sink settings (video output)
    config['sink0'] = {
        'enable': '1',
        'type': '2',  # 2 for RTSP streaming, 3 for file output
        'container': '1',  # 1 for MP4
        'codec': '1',  # 1 for H.264
        'sync': '0',
        'width': '1280',
        'height': '720',
        'bitrate': '4000000',
        'gpu-id': '0'
    }
    
    # Save config file
    with open('deepstream_app_config.txt', 'w') as f:
        config.write(f)
    
    # Create labels file
    with open('labels.txt', 'w') as f:
        f.write("person\n")
        f.write("bicycle\n")
        f.write("car\n")
        # Add more COCO classes as needed
    
    # Create tracker config
    with open('tracker_config.yml', 'w') as f:
        f.write("tracker-type: NvDCF\n")
        f.write("track-input-tensor: yes\n")
        f.write("gpu-id: 0\n")
        f.write("max-age: 30\n")
        f.write("min-hits: 3\n")
        f.write("iou-threshold: 0.5\n")
    
    # Create YOLO config
    with open('yolov8_config.txt', 'w') as f:
        f.write("[property]\n")
        f.write("infer-dims=3;640;640\n")
        f.write("net-scale-factor=0.0039215697906911373\n")
        f.write("network-mode=0\n")
        f.write("offsets=123;117;104\n")
        f.write("num-detected-classes=80\n")
        f.write("parse-bbox-func-name=NvDsInferParseCustomYolo\n")
        f.write("custom-lib-path=/opt/nvidia/deepstream/deepstream-6.2/lib/libnvds_infercustomparser.so\n")

def extract_and_save_face(frame, obj_meta, frame_number):
    """Extract face from person detection and save it"""
    
    object_id = obj_meta.object_id
    
    # Check if we recently saved a face for this ID to avoid duplicates
    current_time = frame_number
    if object_id in last_saved and (current_time - last_saved[object_id]) < FACE_SAVE_INTERVAL:
        return
    
    # Check if we've saved max faces for this ID
    if object_id in saved_count and saved_count[object_id] >= MAX_SAVED_FACES:
        return
    
    # Get object coordinates
    left = int(obj_meta.rect_params.left)
    top = int(obj_meta.rect_params.top)
    width = int(obj_meta.rect_params.width)
    height = int(obj_meta.rect_params.height)
    
    # Extract person ROI
    person_roi = frame[top:top+height, left:left+width]
    if person_roi.size == 0:
        return
    
    # Detect faces in ROI
    try:
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            # Take largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face from ROI
            face = person_roi[y:y+h, x:x+w]
            
            # Save face
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{FACE_SAVE_DIR}/id_{object_id}_{timestamp}_{frame_number}.jpg"
            cv2.imwrite(filename, face)
            print(f"Saved face: {filename}")
            
            # Update tracking dictionaries
            last_saved[object_id] = current_time
            saved_count[object_id] = saved_count.get(object_id, 0) + 1
    except Exception as e:
        print(f"Error saving face: {e}")

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Callback function for sink pad buffer probe to process metadata"""
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
    
    # Retrieve batch metadata from the buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        # Get frame number
        frame_number = frame_meta.frame_num
        
        # Define crossing line y-coordinate based on frame height
        line_y = int(frame_meta.source_frame_height * LINE_Y_POSITION)
        
        # Draw line on frame
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_lines = 1
        line_params = display_meta.line_params[0]
        line_params.x1 = 0
        line_params.y1 = line_y
        line_params.x2 = frame_meta.source_frame_width
        line_params.y2 = line_y
        line_params.line_width = 3
        line_params.line_color.set(1.0, 0.0, 0.0, 1.0)  # Red line
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        # Get frame from video memory
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame = np.array(n_frame, copy=True, order='C')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Process objects in this frame
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            # Only process person class
            if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                object_id = obj_meta.object_id
                
                # Get object's center y-coordinate
                obj_center_y = int(obj_meta.rect_params.top + (obj_meta.rect_params.height / 2))
                
                # Check if object crossed the line
                prev_position = class_tracker.get(object_id, None)
                current_position = obj_center_y
                
                if prev_position is not None:
                    # Check if the object crossed the line from top to bottom
                    if (prev_position < line_y and current_position >= line_y):
                        # Object crossed line, extract and save face
                        extract_and_save_face(frame, obj_meta, frame_number)
                
                # Update position
                class_tracker[object_id] = current_position
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def main():
    """Main function to create and run the DeepStream pipeline"""
    
    # Download model if needed
    download_yolov8_model()
    
    # Create config files
    create_deepstream_config()
    
    # Initialize GStreamer
    Gst.init(None)
    
    # Create the DeepStream pipeline
    pipeline = Gst.Pipeline()
    
    # Create elements
    source = Gst.ElementFactory.make("uridecodebin", "source")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-mux")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    
    # Check if all elements were created successfully
    elements = [pipeline, source, streammux, pgie, tracker, nvvidconv, nvosd, sink]
    for element in elements:
        if not element:
            sys.stderr.write(f"Unable to create {element} element\n")
            sys.exit(1)
    
    # Set properties for source
    source.set_property("uri", "file:///path/to/your/video.mp4")  # Replace with your video path
    
    # Set streammux properties
    streammux.set_property("width", 1280)
    streammux.set_property("height", 720)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 4000000)
    
    # Set pgie properties (primary detector)
    pgie.set_property("config-file-path", "deepstream_app_config.txt")
    
    # Set tracker properties
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-6.2/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "tracker_config.yml")
    tracker.set_property("enable-batch-process", True)
    
    # Add all elements to the pipeline
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    
    # Link elements
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)
    
    # Connect source to streammux
    pad_name = "sink_0"
    sinkpad = streammux.get_request_pad(pad_name)
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux\n")
        sys.exit(1)
    
    # Connect source to pipeline
    source.connect("pad-added", cb_newpad, streammux, sinkpad)
    
    # Add probe to get metadata
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
        sys.exit(1)
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # Start playing
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("Unable to set the pipeline to the playing state\n")
        sys.exit(1)
    
    print("Pipeline is playing...")
    
    # Create a main loop
    loop = GLib.MainLoop()
    
    # Register a signal handler
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_signal, loop)
    
    # Start the main loop
    try:
        loop.run()
    except Exception as e:
        print(f"Error running pipeline: {e}")
    finally:
        # Clean up
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped")

def cb_newpad(decodebin, decoder_src_pad, data, sink_pad):
    """Callback when new source pad is created"""
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    
    if gstname.find("video") != -1:
        decoder_src_pad.link(sink_pad)

def on_signal(loop):
    """Signal handler for Ctrl+C"""
    print("Received interrupt signal, stopping pipeline...")
    loop.quit()
    return True

if __name__ == "__main__":
    import signal
    main()
