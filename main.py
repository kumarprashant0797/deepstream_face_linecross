from asyncore import file_dispatcher
import sys
sys.path.append('../')
import gi
import configparser
import numpy as np
import cv2
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst, GObject
from ctypes import *
import time
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
import uuid
import datetime
import pyds
import os
import signal
import sys

MAX_DISPLAY_LEN=64 
PGIE_CLASS_ID_person = 0

MUXER_OUTPUT_WIDTH=1280
MUXER_OUTPUT_HEIGHT=720
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
pgie_classes_str = ['person']

class_counters = {0: 0}
line_crossing_counts = {"Entry": 0, "Exit": 0}

FACE_SAVE_DIR = os.path.join(os.getcwd(), "line_crossing_faces")
if not os.path.exists(FACE_SAVE_DIR):
    os.makedirs(FACE_SAVE_DIR)
    print(f"Created directory to save faces: {FACE_SAVE_DIR}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_person_face(frame_image, obj_meta, cross_direction):
    if frame_image.shape[2] == 4:
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
    
    left = int(obj_meta.rect_params.left)
    top = int(obj_meta.rect_params.top)
    width = int(obj_meta.rect_params.width)
    height = int(obj_meta.rect_params.height)
    
    person_img = frame_image[top:top+height, left:left+width].copy()
    
    if width < 20 or height < 20 or person_img.size == 0:
        print(f"Person region too small for ID {obj_meta.object_id}, skipping")
        return
    
    gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        print(f"No face detected in person crop for ID {obj_meta.object_id}, trying full frame detection")
        
        margin = int(min(width, height) * 0.2)
        expanded_left = max(0, left - margin)
        expanded_top = max(0, top - margin)
        expanded_width = min(width + 2*margin, frame_image.shape[1] - expanded_left)
        expanded_height = min(height + 2*margin, frame_image.shape[0] - expanded_top)
        
        expanded_person_img = frame_image[expanded_top:expanded_top+expanded_height, 
                                         expanded_left:expanded_left+expanded_width].copy()
        
        expanded_gray = cv2.cvtColor(expanded_person_img, cv2.COLOR_BGR2GRAY)
        
        expanded_faces = face_cascade.detectMultiScale(
            expanded_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in expanded_faces:
            faces.append((x + expanded_left - left, y + expanded_top - top, w, h))
    
    if len(faces) == 0:
        print(f"No face detected for Person ID {obj_meta.object_id}, saving top 1/3 of person image")
        
        top_third_height = int(height / 3)
        face_img = person_img[0:top_third_height, 0:width]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{cross_direction}_id{obj_meta.object_id}_top_third.jpg"
    else:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        x = max(0, min(x, person_img.shape[1] - 1))
        y = max(0, min(y, person_img.shape[0] - 1))
        w = min(w, person_img.shape[1] - x)
        h = min(h, person_img.shape[0] - y)
        
        face_img = person_img[y:y+h, x:x+w]
        print(f"Face detected for Person ID {obj_meta.object_id}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{cross_direction}_id{obj_meta.object_id}_face.jpg"
    
    filepath = os.path.join(FACE_SAVE_DIR, filename)
    
    cv2.imwrite(filepath, face_img)
    print(f"Saved {'face' if len(faces) > 0 else 'top third'} image for ID {obj_meta.object_id} crossing {cross_direction} line to {filepath}")

def get_nvds_buf_surface(gst_buffer, batch_id):
    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
    frame_image = np.array(n_frame, copy=True, order='C')
    return frame_image

def nvanalytics_src_pad_buffer_probe(pad, info, u_data):
    global class_counters
    global line_crossing_counts

    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()

    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        class_counters = {0: 0}
        
        frame_image = get_nvds_buf_surface(gst_buffer, frame_meta.batch_id)

        DetectedClass = None
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == 0:
                DetectedClass = pgie_classes_str[obj_meta.class_id]
                class_counters[0] += 1

            l_user_meta = obj_meta.obj_user_meta_list
            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                except StopIteration:
                    break

                try:
                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                        
                        if user_meta_data.lcStatus:
                            line_crossed = []
                            for status_string in user_meta_data.lcStatus:
                                if status_string:
                                    line_crossed.append(status_string)
                            
                            for line_label in line_crossed:
                                if "Exit" in line_label:
                                    line_crossing_counts["Exit"] += 1
                                    print(f"Person ID {obj_meta.object_id} crossed Exit line")
                                    save_person_face(frame_image, obj_meta, "Exit")
                                    
                                elif "Entry" in line_label:
                                    line_crossing_counts["Entry"] += 1
                                    print(f"Person ID {obj_meta.object_id} crossed Entry line")
                                    save_person_face(frame_image, obj_meta, "Entry")
                                    
                                else:
                                    print(f"Person ID {obj_meta.object_id} crossed line: {line_label}")
                                    save_person_face(frame_image, obj_meta, line_label)
                        
                        if hasattr(user_meta_data, 'dirStatus') and user_meta_data.dirStatus:
                            dir_labels = []
                            for status_string in user_meta_data.dirStatus:
                                if status_string:
                                    dir_labels.append(status_string)
                            
                            for direction_label in dir_labels:
                                print(f"Person ID {obj_meta.object_id} moving in direction: {direction_label}")
                except Exception as e:
                    print(f"Error processing metadata: {e}")

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        if class_counters[0] == 0:
            DetectedClass = "None"
        else:
            DetectedClass = "person"

        print("Frame Number=", frame_number, "stream id=", frame_meta.pad_index, 
              "Number of Objects=", num_rects, "Persons detected:", class_counters[0],
              "Detected Class:", DetectedClass, 
              "Line crossings - Entry:", line_crossing_counts["Entry"], "Exit:", line_crossing_counts["Exit"])

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    print("gstname=",gstname)
    if(gstname.find("video")!=-1):

        print("features=",features)
        if features.contains("memory:NVMM"):
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

def create_source_bin(index,uri):
    print("Creating source bin")

    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")


    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)


    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args):
    if len(args) < 2:
        print("Usage: python3 code.py <RTSP_URL_1> <RTSP_URL_2> ...")
        sys.exit(1)

    sources = args[1:]
    number_sources = len(sources)
  
    GObject.threads_init()
    Gst.init(None)

    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    print("Creating streamux \n ")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)

    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=sources[i]
        if uri_name.find("file://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
  
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 400000)
    streammux.set_property('attach-sys-ts', True)
    streammux.set_property('compute-hw',1)
    streammux.set_property('live-source',1)
    
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")

    config = configparser.ConfigParser()
    config.read('tracker_config.txt')
    
    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu-id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
    
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    nvanalytics.set_property("config-file", "config_nvdsanalytics.txt")
    
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    filter1.set_property("caps", caps1)

    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property('sync', 0)
    sink.set_property('qos', 0)
        
    pgie.set_property('config-file-path', "./config_pgie.txt")

    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvconv1)
    pipeline.add(filter1)
    pipeline.add(nvanalytics)
    pipeline.add(tiler)
    pipeline.add(nvosd)
    pipeline.add(sink)

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(nvconv1)
    nvconv1.link(filter1)
    filter1.link(queue3)
    queue3.link(nvanalytics)
    nvanalytics.link(queue4)
    queue4.link(tiler)
    tiler.link(queue5)
    queue5.link(nvosd)
    nvosd.link(sink)
  
    print("Linking elements in the Pipeline \n")

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    nvanalytics_src_pad=nvanalytics.get_static_pad("src")
    if not nvanalytics_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        nvanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)

    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
