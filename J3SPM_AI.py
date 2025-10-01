## Version01,release 01
## pytorch 2.4 version <=> hbconf.py 분리함.
## Segmentation default model 처리되도록 함.

import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QTextEdit, QPushButton, QVBoxLayout, QWidget, QToolButton, QFileDialog, QMessageBox

from mpl_toolkits.mplot3d import Axes3D
import subprocess
from J3SPM_AI_GUI import Ui_MainWindow
from datetime import datetime 
from Dialog_1 import MyDialog
from Worker_class import Worker
import cv2
import pandas as pd
import os
import random
import shutil
import yaml
import datetime

import serial
from serial.tools import list_ports
import socket
import struct

import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
from utils.downloads import attempt_download

current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_dir = os.path.join(current_dir, '..', 'yolov5_J3SPM')

sys.path.append(yolov5_dir)

from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    scale_segments,
    xyxy2xywh,
)
import csv
import platform

from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

from labelme2yolov5 import Labelme2YOLO
import webbrowser

class SerialThread(QThread):
    received_signal = pyqtSignal(str)

    def __init__(self):#, port, baudrate=9600, parent=None):
        super(SerialThread, self).__init__()
        self.serial = None
        self.SPM_port = None
        self.serial_connection = None
        self.tcp_connection = True
        self.baud_rate = 9600
        self.running = True
        self.client = None #socket.socket(socket.AF_INET, socket.SOCK_STREAM)\
        self.ethernet = False

    def run(self):
        while self.running:
            if self.ethernet:
                if self.tcp_connection==None:
                    line = self.client.recv(1024).decode()
                    if line.startswith('3'):
                        self.received_signal.emit(line.strip())
            else:
                if self.serial_connection.isOpen():
                    line = self.serial_connection.readline().decode('utf-8')
                    if line.endswith('\n'):
                        self.received_signal.emit(line.strip())

    
    def connect_to_SPM(self, SPM_port, baud_rate=9600):
        message=""
        if SPM_port:
            try:
                self.serial_connection = serial.Serial(SPM_port, baud_rate, timeout=1)
                message=f"Connected to SPM on {SPM_port} at {baud_rate} baud."
            except Exception as e:
                message=f"Failed to connect to SPM on {SPM_port}: {e}"
        else:
            message="No SPM device is set for connection."
        
        return message
    
    def connect_tcp_SPM(self, host, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        message=""
        try:
            server_address = (host, int(port))
            self.tcp_connection=self.client.connect(server_address)
            message=f"Connected to SPM on {host} at port {port}."

        except Exception as err:
            message=f"Failed to connect to SPM (TCP/IP) : {err}"
        
        return message

    def send_string_to_SPM(self, cb_tcp, string_data):
        #global ethernet
        if self.serial_connection and self.serial_connection.isOpen() or (self.tcp_connection==None):
            try:
                if cb_tcp:
                    self.client.sendall(string_data.encode())  # TCP/IP
                    formattedText=f"Sent to SPM: {string_data}"
                    self.ethernet = True
                else:
                    self.serial_connection.write(string_data.encode())  # Serial: Send string to SPM
                    formattedText=f"Sent to SPM: {string_data}"
                    self.ethernet = False
                
            except Exception as e:
                formattedText=f"Failed to send data to SPM: {e}"
                
        else:
            formattedText=f"Serial connection not established. Please connect first."
        return formattedText    
        
    def stop(self):
        self.running = False
        if self.serial_connection and self.serial_connection.isOpen():
            self.serial_connection.close()
        elif self.tcp_connection == None:
            self.client.close()
        else:
            return None

class MainWindow(QMainWindow, Ui_MainWindow, Labelme2YOLO):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

        self.SPM_finder = SerialThread()
        self.SPM_finder.received_signal.connect(self.handle_data)
  

        self.linedata=[]
        self.zarea = []
        self.converted_list =[]
        self.lines_received = 0
        self.img_received =0
        self.rectangles = []
        self.line_spacing =[]
        self.image,self.folder_path =None, None
        self.COM=0

        self.imgname =''
        self.mdname = ''
        self.yamlname =''
        self.hypername =''
        self.results_df = pd.DataFrame()
        self.det_df = pd.DataFrame()
        self.results = None
        self.detectedobj = 0
        self.loaded_result = None
        self.imWidth, self.imHeight =0,0
        self.scandone = False
        self.refreshPorts()
        self.pB_SerConect.setEnabled(False)
        self.serialConnection = None
        self.SPM_finder.running = False
        self.SPM_finder.start()
        self.filetype =''


    def handle_data(self):
        self.zscan_obj()
   
    def closeEvent(self, event):
         self.SPM_finder.stop()  # stopping serial communication thread 
         super().closeEvent(event)
    

    def stopThread(self):
        if self.SPM_finder.isRunning():
            self.SPM_finder.requestInterruption()
            self.SPM_finder.quit()
            self.SPM_finder.wait()

         
    def dialog_box(self,note):
        dialog=MyDialog(note)
        dialog.exec_()

    def do_gwyddion(self):
        program_path = 'C:\\Program Files\\Gwyddion\\bin\\gwyddion.exe'
        try:
            # Gwyddion run
            subprocess.run([program_path], check=True)
        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "Can't find Gwyddion. Select file by yourself.")
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Gwyddion.exe",
                "",  
                "exe file (*.exe);;all files (*)"
            )
            
            if file_path:  
                subprocess.run([file_path])
            else:
                return
    
    def do_labelstudio(self):
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        script_path = os.path.join(current_dir, 'labelimg_J3SPM', 'labelImg.py')
        python_executable = 'python'
        subprocess.run([python_executable, script_path], check=True)


    def do_labelme(self):
        file_path = 'labelme'
        subprocess.run([file_path])

    def do_labelme2yolo(self):
        json_directory = QFileDialog.getExistingDirectory(self,"Select json Folder")
        if not json_directory:
            QMessageBox.warning(self, "Warning", "No directory selected. Operation cancelled.")
            return None
        else:
            labelme_to_yolo = Labelme2YOLO(json_dir=json_directory)
            labelme_to_yolo.convert(val_size=float(self.lE_ValidR.text())*0.01,test_size=float(self.lE_TestR.text())*0.01)
            self.dialog_box("Converting completed.")

    
    def do_composedset(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,"Select Folder")
        destination_folder = os.path.join(self.folder_path, "raw_data")

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            
            if os.path.isfile(file_path):
                shutil.copy(file_path, destination_folder)
        
        if self.folder_path:
            img_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg','.PNG','.JPG','.webp'))]
            txt_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.txt')]
            
            if not img_files or not txt_files:
                self.dialog_box("No image/label files found in the directory.")
                return
            file_pairs = [(img, os.path.splitext(img)[0] + '.txt') for img in img_files if os.path.splitext(img)[0] + '.txt' in txt_files]
            if not file_pairs:
                self.dialog_box("No matching image-text file pairs found in the directory.")
                return
            random.shuffle(file_pairs)
            
            total = len(file_pairs)
            train_end = int(total * float(self.lE_TrainR.text())*0.01)
            test_size = max(1, int(total * float(self.lE_TestR.text())*0.01))
            test_end = train_end + test_size
            self.move_files(self.folder_path, file_pairs[:train_end], 'train')
            self.move_files(self.folder_path, file_pairs[train_end:test_end], 'test')
            self.move_files(self.folder_path, file_pairs[test_end:], 'valid')

            with open(os.path.join(self.folder_path,'classes.txt'), 'r') as file:
                classes = [line.strip() for line in file.readlines()]
                print(classes)
                        
                dataset_dict = {
                    'train': os.path.normpath(os.path.join(self.folder_path, 'train')),  
                    'val': os.path.normpath(os.path.join(self.folder_path, 'valid')),      
                    'test': os.path.normpath(os.path.join(self.folder_path, 'test')),   
                    'nc': len(classes),   
                    'names': classes     
                }

            with open(os.path.join(self.folder_path,'data.yaml'), 'w') as file2:
                
                yaml.dump(dataset_dict, file2, sort_keys=False)
                self.tBR_Dataset.setText(yaml.dump(dataset_dict))
                self.dialog_box("Dataset composed successfully")

        else:
            self.dialog_box("Dataset composition cancelled")
         

    def move_files(self,folder_path, file_pairs, destination):
        images_dest = os.path.join(folder_path, destination, 'images')
        labels_dest = os.path.join(folder_path, destination, 'labels')
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)

        for img_file, txt_file in file_pairs:
            shutil.move(img_file, images_dest)
            shutil.move(txt_file, labels_dest)


    def do_opentopo(self)    :
        filePath, _ = QFileDialog.getOpenFileName(None, "Open Picture", "", "Image Files (*.png *.jpg *.bmp)")
        if filePath :
            image=cv2.imread(filePath)
            fileNameWithExtension = os.path.basename(filePath)
            creation_time = os.path.getctime(filePath)
            modification_time = os.path.getmtime(filePath)

            creation_date = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
            modification_date = datetime.datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d %H:%M:%S')
            rgb_image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.show()
            height,width=image.shape[:2]
            contents=f'======================\nFilename: {fileNameWithExtension}\nImage size: {width}x{height} pixels'
            self.tBR_Imageinfo.append(contents)
            contents=f'Creation date: {creation_date}\nModifi. date: {modification_date}'
            self.tBR_Imageinfo.append(contents)
        
    def do_training(self):
        if self.do_combobox_3()=='detection':
            command = ['python', 'train.py']
        else:
            command = ['python', './segment/train.py']

        if self.sB_IPixes.value() > 0:
            command.extend(['--img', str(self.sB_IPixes.value())])
        if self.sB_Batch.value() > 0:
            command.extend(['--batch', str(self.sB_Batch.value())])
        if self.sB_Epochs.value() > 0:
            command.extend(['--epochs', str(self.sB_Epochs.value())])
        if self.yamlname != '':
            command.extend(['--data', self.yamlname])

        if self.do_combobox_3()=='detection':
            command.extend(['--weights', self.do_combobox() + '.pt'])
            command.extend(['--cfg', './models/'+self.do_combobox_2() + '.yaml'])
        else:
            command.extend(['--weights', self.do_combobox() + '-seg.pt'])
            command.extend(['--cfg','./models/segment/' + self.do_combobox_2() +'-seg.yaml'])
            
        if self.hypername != '':
            command.extend(['--hyp', self.hypername])
        if self.cB_Cache.isChecked():  
            command.append('--cache')
        if self.lE_Project.text() != '':
            command.extend(['--project', self.lE_Project.text()])
        if self.lE_Model.text() != '':
            command.extend(['--name', self.lE_Model.text()])
        if self.lineEdit_3.text() !='':
            command.extend(self.lineEdit_3.text().split())

        print(command)
        self.displayFormattedText(self.tBR_Train,'Tranining...')
        self.worker = Worker(command)
        self.worker.update_text.connect(self.appendText)
        self.worker.update_error.connect(self.appendError)
        self.worker.start()
    
    def do_coLab(self):
        webbrowser.open('https://youtu.be/YfryRAA26ZE')

    def appendText(self, text):
        self.tBR_Train.append(text)

    def appendError(self, text):
        self.tBR_Train.append(f"<font color='red'>{text} </font>")
        
    def do_modelselect(self):
        self.mdname = self.openFileNameDialog(2)
        filename = os.path.basename(self.mdname)
        self.lB_modelname.setText(filename)        
        return None
    
    def create_unique_directory(self,base_path):
        folder_index = 0
        current_path = base_path
        while os.path.exists(current_path):
            folder_index += 1
            current_path = f"{base_path}{folder_index}"
        os.makedirs(current_path)
        return current_path

    def do_inference(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.chdir(yolov5_dir)

        if self.mdname=='':
                model = torch.hub.load(yolov5_dir, 'custom', path='./models/yolov5s.pt', source='local')
                self.mdname = './models/yolov5s.pt'
        else:
                model = torch.hub.load('.', 'custom', path=self.mdname, source='local')

        if self.imgname == '':
            self.imgname = './testimg.jpg'
            
        if self.do_combobox_4()=='detect':
            if self.filetype == 'Image':
                self.infer_code(device)
            elif self.filetype == 'Video':
                self.infer_code(device)
                return

        else:                               # segmentation
            from utils.downloads import attempt_download

            fn = Path(self.mdname).name if isinstance(self.mdname, str) else ''
            if (not fn) or ('-seg' not in fn):
                # 사용자가 모델을 선택 안 했거나 탐지용을 잡아온 경우 → seg로 교체
                try:
                    # 사용자 기본 크기와 맞추려면 s/m/l/x를 추론해 치환해도 되지만
                    # 지정이 없을 때는 s로 통일
                    self.mdname = attempt_download('yolov5s-seg.pt')
                except Exception:
                    # 네트워크/권한 문제 등일 때도 s-seg로 한 번 더 시도
                    self.mdname = './yolov5s-seg.pt'  # 로컬에 있을 수도 있음

            self.segment_code(device, model)
            self.segment_code(device,model)



    def infer_script(self):
        command = ['python', './detect.py']
        command.extend(['--weights', self.mdname])
        command.extend(['--source', self.imgname])
        command.extend('--save-txt --view-img --conf-thres 0.7'.split())
        print(command)
        subprocess.run(command)

    def infer_code(self,device):
        columns = ["Image Name", "Class ID", "Class Name", "Confidence", "x_min", "y_min", "x_max", "y_max"]
        results_df = pd.DataFrame(columns=columns)

        imgsz=(640,640)  # inference size (height, width)
        view_img=self.cB_viewImg.isChecked()
        save_txt=self.cB_saveTxt.isChecked()
        save_csv=self.cB_saveCsv.isChecked()
        save_conf=self.cB_saveConf.isChecked()
        save_crop=self.cB_saveCrop.isChecked()
        nosave=self.cB_noSave.isChecked()
        agnostic_nms=self.cB_agnoNms.isChecked()
        augment=self.cB_augment.isChecked()
        visualize=self.cB_visualize.isChecked()
        update=self.cB_update.isChecked()
        exist_ok=self.cB_existOk.isChecked()
        hide_labels=self.cB_hideLab.isChecked()
        hide_conf=self.cB_hideConf.isChecked()
        half=self.cB_half.isChecked()
        dnn=self.cB_dnn.isChecked()
        device=self.lE_device.text()
        conf_thres=self.dB_confThres.value()
        iou_thres=self.dB_ioufThres.value()
        max_det=self.sB_maxDet.value()
        project=self.lE_prjname.text()
        name=self.lE_prjname_2.text()
        vid_stride=self.sB_vidStride.value()
        line_thickness=self.sB_lineThick.value()
        if not self.lE_classes.text().strip():
            classes=None
        else:
            class_indices = self.lE_classes.text().split()
            classes=[int(index) for index in class_indices]

        save_img = not nosave and not self.imgname.endswith(".txt")  # save inference images

        # Directories
        device = select_device(device)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        #Load model
        model = DetectMultiBackend(self.mdname, device=device, dnn=dnn, data='', fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        
        bs=1
        dataset = LoadImages(self.imgname, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        #Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        results_dect = []
        log_messages = []

        for path, im, im0s, vid_cap, s in dataset:
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)
            log_message = f"{path}: {im.shape[2]}x{im.shape[3]} "
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            self.imWidth, self.imHeight =im.shape[2], im.shape[3]
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                """Writes prediction data for an image to a CSV file, appending if the file exists."""
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not csv_path.is_file():
                        writer.writeheader()
                    writer.writerow(data)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if False else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        if save_csv:
                            write_to_csv(p.name, label, confidence_str)

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            if self.filetype == 'Video':
                                self.displayFormattedText(self.tBR_Result2,xywh)

                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                if True:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            log_message += f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
            log_messages.append(log_message)
            #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            for log_message in log_messages:
                LOGGER.info(log_message)

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if True else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update: 
            strip_optimizer(self.mdname[0])  # update model (to fix SourceChangeWarning) 
        
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x_min, y_min, x_max, y_max = xyxy
                cls_id = int(cls)
                cls_name = names[cls_id]
                confidence = float(conf)
                result = {
                    "class ID": cls_id,
                    "class": cls_name,
                    "confidence": confidence,
                    "xmin": int(xyxy[0].item()),
                    "ymin": int(xyxy[1].item()),
                    "xmax": int(xyxy[2].item()),
                    "ymax": int(xyxy[3].item())
                }
                results_dect.append(result)
                        # Convert list of dicts to DataFrame and concatenate
        
        self.results_df = pd.DataFrame(results_dect)
            

        # Save the DataFrame to CSV file at the end of detection
        results_txt_path = save_dir / "detection_results.txt"
        self.results_df.to_csv(results_txt_path, index=False, sep='\t')
        self.displayFormattedText(self.tBR_Result1,log_messages)
        self.dialog_box('Inference completed.')

    def segment_code(self, device, model):
        imgsz = (640, 640)  # inference size (height, width)
        view_img = self.cB_viewImg.isChecked()
        save_txt = self.cB_saveTxt.isChecked()
        save_csv = self.cB_saveCsv.isChecked()
        save_conf = self.cB_saveConf.isChecked()
        save_crop = self.cB_saveCrop.isChecked()
        nosave = self.cB_noSave.isChecked()
        agnostic_nms = self.cB_agnoNms.isChecked()
        augment = self.cB_augment.isChecked()
        visualize = self.cB_visualize.isChecked()
        update = self.cB_update.isChecked()
        exist_ok = self.cB_existOk.isChecked()
        hide_labels = self.cB_hideLab.isChecked()
        hide_conf = self.cB_hideConf.isChecked()
        half = self.cB_half.isChecked()
        dnn = self.cB_dnn.isChecked()
        device = self.lE_device.text()
        conf_thres = self.dB_confThres.value()
        iou_thres = self.dB_ioufThres.value()
        max_det = self.sB_maxDet.value()
        project = self.lE_prjname.text() + '-seg'
        name = self.lE_prjname_2.text() + '-seg'
        vid_stride = self.sB_vidStride.value()
        line_thickness = self.sB_lineThick.value()
        retina_masks = self.cB_retina.isChecked()
        if not self.lE_classes.text().strip():
            classes = None
        else:
            class_indices = self.lE_classes.text().split()
            classes = [int(index) for index in class_indices]
        save_img = not nosave and not self.imgname.endswith(".txt")  # save inference images

        # Directories
        device = select_device(device)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # model = DetectMultiBackend(self.mdname, device=device, dnn=dnn, data='', fp16=half)

       # --- SEG weights auto-resolve & auto-download --------------------------------
        weights_in = self.mdname[0] if isinstance(self.mdname, (list, tuple)) else self.mdname
        p = Path(str(weights_in)).expanduser()

        def _force_to_seg(name: str) -> str:
            # yolov5s.pt -> yolov5s-seg.pt
            return name.replace('.pt', '-seg.pt') if name.endswith('.pt') and '-seg' not in name else name

        if p.exists():
            # ★ 파일이 있어도 탐지용이면 seg로 교체 + 다운로드(없으면 받기)
            if '-seg' not in p.name:
                LOGGER.warning(f"WARNING ⚠️ '{p.name}'는 탐지 전용으로 보입니다. 세그멘테이션 가중치('-seg')로 교체합니다.")
                try:
                    seg_weights = attempt_download(_force_to_seg(p.name))
                except Exception:
                    seg_weights = attempt_download('yolov5s-seg.pt')
            else:
                seg_weights = str(p)
        else:
            # 없으면 이름 교정 후 다운로드
            name = p.name or str(weights_in) or "yolov5s-seg.pt"
            name = _force_to_seg(name)
            try:
                seg_weights = attempt_download(name)
            except Exception:
                seg_weights = attempt_download("yolov5s-seg.pt")
        # -----------------------------------------------------------------------------

        # Load model (세그 가중치 경로로 강제)
        model = DetectMultiBackend(seg_weights, device=device, dnn=dnn, data='', fp16=half)

        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        bs = 1
        dataset = LoadImages(self.imgname, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        results_seg = []
        log_messages = []
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, proto = model(im, augment=False, visualize=visualize)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            # Process predictions
            log_message = f"{path}: {im.shape[2]}x{im.shape[3]} "
            self.imWidth, self.imHeight =im.shape[2], im.shape[3]
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                s = ""  # Reset s for each image

                imc = im0.copy() if False else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    if retina_masks:
                        # scale bbox first then crop masks
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Segments
                    if save_txt:
                        segments = [
                            scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                            for x in reversed(masks2segments(masks))
                        ]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Collect results for DataFrame
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        result = {
                            "class": names[int(cls)],
                            "confidence": conf.item(),
                            "xmin": int(xyxy[0].item()),
                            "ymin": int(xyxy[1].item()),
                            "xmax": int(xyxy[2].item()),
                            "ymax": int(xyxy[3].item())
                        }
                        results_seg.append(result)

                    # Log message for current image
                    log_message += f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
                    log_messages.append(log_message)

                    # Mask plotting
                    annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() / 255
                        if retina_masks
                        else im[i],
                    )

                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_txt:  # Write to file
                            seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == "Linux" and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                        exit()


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            for log_message in log_messages:
                LOGGER.info(log_message)

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if True else ""
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(self.mdname[0])  # update model (to fix SourceChangeWarning)

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results_seg)
        # Save DataFrame to a text file
        txt_output_path = save_dir / f"results_{name}.txt"
        self.results_df.to_csv(txt_output_path, index=False, sep='\t')

        self.displayFormattedText(self.tBR_Result1,log_messages)
        #print(self.results_df)

        
    def do_search(self):
        if self.results_df.empty :
            self.dialog_box('No inference results, Infer first.')
            return
        
        target_class_name = self.lE_Class.text()  # 필터링하려는 클래스 이름
        
        if 'name' in self.results_df.columns:
            self.results_df['target'] = self.results_df['name']
        elif 'class' in self.results_df.columns:
            self.results_df['target'] = self.results_df['class']
        filtered_df = self.results_df[self.results_df['target'] == target_class_name]

        #filtered_df = self.results_df[self.results_df['name'] == target_class_name]
        filtered_df=filtered_df[filtered_df['confidence'] > float(self.lE_Confi.text())*0.01]

        if filtered_df.empty:
            target_class_name = target_class_name + " was not detected!!"
            self.dialog_box(target_class_name)
            return

        coordinates = filtered_df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].values
        co0 = coordinates.tolist()

        self.detectedobj = len(co0)
        self.converted_list = [[int(item) for item in sublist] for sublist in co0]
        self.lB_ClassQu.setText(str(self.detectedobj))

        self.displayFormattedText2(self.tBR_Result2, self.converted_list)
        
    def do_openimg(self):
        self.imgname = self.openFileNameDialog(1)
        filename = os.path.basename(self.imgname)
        self.filetype = self.determine_file_type(filename)
        if self.filetype == "Image":

            self.lB_imagname.setText(filename)    
            if self.imgname:
                imageo=Image.open(self.imgname)
                imageo.show()
            else:
                return
        elif self.filetype == "Video":
            self.lB_imagname.setText(filename)    
            if self.imgname:
                self.open_video_with_default_player(self.imgname)
                
            else:
                return

    def open_video_with_default_player(self, file_path):
        # Check operating system
        if sys.platform == 'win32':
            subprocess.run(['start', file_path], shell=True)
        elif sys.platform == 'darwin':
            subprocess.run(['open', file_path])
        elif sys.platform == 'linux':
            subprocess.run(['xdg-open', file_path])
        else:
            print("Unsupported OS")

    def play_video(self,file_path):
        # Open video file
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Play video
        while True:
            ret, frame = cap.read()
            if not ret:
                if last_frame is not None:
                    frame = last_frame  # Use the last valid frame if reading a new frame fails
                else:
                    break  # End loop if reading fails from the first frame
            else:
                last_frame = frame  

            cv2.imshow('Video Playback', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):  # 'q' play stop
                break

        # Exit handling
        cap.release()  
        cv2.destroyAllWindows()  

    def determine_file_type(self,filename):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

        _, ext = os.path.splitext(filename)
        ext = ext.lower()  

        if ext in image_extensions:
            return "Image"
        elif ext in video_extensions:
            return "Video"
        else:
            return "Unknown"

    def do_dataload(self):
        self.yamlname = self.openFileNameDialog(3)
        if self.yamlname :
            self.lB_DYaml.setText(self.yamlname)
        else:
            return
    
    def do_hyperload(self):
        self.hypername = self.openFileNameDialog(4)
        if self.hypername:
            self.lB_Hyper.setText(self.hypername)
        else:
            return  

    def do_combobox(self):
        text=self.comboBox.currentText()
        return text.lower()
    
    def do_combobox_2(self):
        text=self.comboBox_2.currentText()
        return text.lower()
    
    def do_combobox_3(self):
        text=self.comboBox_3.currentText()    
        return text.lower()
    
    def do_combobox_4(self):
        text=self.comboBox_4.currentText()
        if text.lower() == 'detect':
            self.pB_Saveres.setEnabled(False)
            self.pB_Loadres.setEnabled(False)
        else:
            self.pB_Saveres.setEnabled(False)
            self.pB_Loadres.setEnabled(False)
        return text.lower()   
    
    def do_zscan(self):
        self.img_received = 0
        self.pBA_ProgFS.setValue(0)
        if self.converted_list ==[]:
            self.displayFormattedText1(self.tBR_Serialput,'No searching data, Search first.')
            return
        if self.SPM_finder.serial_connection or self.cB_Sim.isChecked() or (self.SPM_finder.tcp_connection==None):
            if self.cB_Allobj.isChecked():
                if self.cB_Sim.isChecked(): ##multi sim 
                    self.pBA_ProgFS.setRange(0,self.detectedobj)
                    for i in range(self.detectedobj): 
                        self.zarea = self.converted_list[i]
                        self.print_info()
                        self.lB_Currentob.setText(str(self.img_received+1))
                        self.pBA_ProgFS.setValue(self.img_received+1)   
                        self.img_received += 1
                    self.draw_scanboxes()

                    return
                
                else: #multi realscan
                    self.zarea=self.converted_list[0]       
                    self.zscan_obj()            
                    try:
                        self.SPM_finder.running = True
                    except KeyboardInterrupt:
                        print("Scan 취소")

            else:   ## single scan
                if self.sB_indexZ.value() > self.detectedobj-1 : 
                    self.dialog_box('Index is larger than number of detected image')
                    return

                if self.cB_Sim.isChecked():     #single sim
                    self.zarea=self.converted_list[self.sB_indexZ.value()]
                    self.draw_scanbox()
                    self.print_info()
                    return
                else:                           #single realscan
                    self.zarea=self.converted_list[self.sB_indexZ.value()]
                    self.zscan_obj()
                    try:
                        self.SPM_finder.running = True
                    except KeyboardInterrupt:
                        print("Scan 취소")
        else:
            self.displayFormattedText1(self.tBR_Serialput,"No connection, Connection first.")


    def zscan_obj(self):
        if(self.img_received == self.detectedobj):
                self.SPM_finder.running = False
                self.dialog_box('Scans completed')
                return None
        self.lB_Currentob.setText(str(self.img_received+1))

        if self.cB_Allobj.isChecked()==True:
            self.zarea=self.converted_list[self.img_received]
            self.pBA_ProgFS.setRange(0,self.detectedobj)
            self.pBA_ProgFS.setValue(self.img_received+1)   
            self.draw_scanbox()
            self.print_info()
            try:
                self.SPM_finder.running = True
                self.SPM_finder.start()
            except KeyboardInterrupt:
                print("Scan cancelled")
        else:
            self.SPM_finder.running = False
            self.zarea=self.converted_list[self.sB_indexZ.value()]
            self.pBA_ProgFS.setRange(0,1)  
            self.draw_scanbox()
            self.img_received=0
            self.lines_received=0
            self.SPM_finder.running = False
            self.print_info()

    def do_zscancancel(self):
        self.SPM_finder.running = False

    def draw_scanbox(self):
        self.image=cv2.imread(self.imgname)

        self.rectangles = [self.zarea]
        heights=[rect[3]-rect[1] for rect in self.rectangles]
        self.line_spacing = [int(height / (self.sB_LinesZ.value()-1)) 
                        if int(height / (self.sB_LinesZ.value())-1)>2 else 2 for height in heights]
        
        self.draw_raster_lines()

        rgb_image= cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.show()

    def draw_scanboxes(self):
        self.image=cv2.imread(self.imgname)
        for i in range(self.detectedobj): 
            self.zarea = self.converted_list[i]
            self.rectangles = [self.zarea]
            heights=[rect[3]-rect[1] for rect in self.rectangles]
            self.line_spacing = [int(height / (self.sB_LinesZ.value()-1)) 
                                 if int(height / (self.sB_LinesZ.value())-1)>2 else 2 for height in heights]
            self.draw_raster_lines()
        rgb_image= cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.show()   


    def print_info(self):
        scan_lines = self.sB_LinesZ.value()
        Res_X = self.sB_ReXZ.value()
        scaled=self.pixel2length(self.zarea)
        scalev=self.length2volt(scaled)
        
        index = self.img_received if self.cB_Allobj.isChecked() else self.sB_indexZ.value()

        data_str = "{:10},{:03d},{:03d},{:03d},{:5.3f},{:5.3f},{:5.3f},{:5.3f}\n".format(self.lE_Class.text(), index, Res_X, scan_lines, scalev[0], scalev[1], scalev[2], scalev[3])

        if self.cB_Sim.isChecked():
            self.displayFormattedText1(self.tBR_Serialput,f'Sim: {data_str}')

        else:
            message=self.SPM_finder.send_string_to_SPM(self.cB_Ethernet.isChecked(), data_str)
            self.displayFormattedText1(self.tBR_Serialput,message) 
           
            if self.cB_Allobj.isChecked():
                self.img_received +=1


    def do_saveres(self):
        self.resname = self.openFileNameDialog(6)
        print(self.resname)
        if self.resname :
            self.drawsave_results()
            self.dialog_box('Results are saved successfully!')
        else:
            self.dialog_box('Results saving is cancelled')
        
    def do_loadres(self):
        self.resname = self.openFileNameDialog(5)
        if self.resname :
            self.results_df =pd.read_csv(self.resname, sep=' ')
            self.displayFormattedText(self.tBR_Result1,self.results_df)
            
            name_folderPath, ext = os.path.splitext(self.resname)
            jpg_path = f"{name_folderPath}.jpg"
            self.imgname=jpg_path

            imageo=Image.open(jpg_path)
            imageo.show()
            self.imWidth, self.imHeight =imageo.size
            
        else:
            self.dialog_box('Results loading is cancelled')

    def openFileNameDialog(self,choice):
        options = QFileDialog.Options()
        if choice ==1 :
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Data open",
                                                  "",
                                                  "image files (*.jpg *.jpeg *.png *.bmp *.gif);;movie files(*.avi *.mp4 *.mov *.mkv);;all files (*)",
                                                  options=options)
        elif choice ==2:
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Model open",
                                                  "",
                                                  "model file (*.pt);;all files (*)",
                                                  options=options)
        elif choice ==3:
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Dataset Yaml Open",
                                                  "",
                                                  "Dataset Yaml file (*.yaml);;all files (*)",
                                                  options=options)
        elif choice ==4:
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Hyper Yaml Open",
                                                  "",
                                                  "Hyper Yaml file (*.yaml);;all files (*)",
                                                  options=options)
        elif choice ==5:
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Result Open",
                                                  "",
                                                  "Text file (*.txt);;all files (*)",
                                                  options=options)
        elif choice ==6:
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File", "",
                                                  "JPG Files (*.jpg)", options=options)
            if fileName :
                if not fileName.endswith('.jpg'):
                    fileName += '.jpg'
                else:
                    return None

        elif choice ==7:
            fileName = QFileDialog.getExistingDirectory(self,"Select Folder")

        return fileName

    def displayFormattedText(self, QTextBrower, strings):
        formattedText = f"{strings}"
        QTextBrower.append(formattedText)
        QTextBrower.append('-------------------------------')

    def displayFormattedText1(self, QTextBrower, strings):
        formattedText = f"{strings}"
        QTextBrower.append(formattedText)
    
    def displayFormattedText2(self, QTextBrower, strings):
        formattedText = "\n".join(str(sublist) for sublist in strings)
        QTextBrower.append(self.lE_Class.text())
        QTextBrower.append(formattedText)

    def draw_raster_lines(self, rect_color=(20, 20, 255), line_color=(0, 122, 0), line_thickness=2):

        i=0
        for rect in self.rectangles:
            center_x, center_y, width, height = rect
            top_left = (int(center_x ), int(center_y))
            bottom_right = (int(width), int(height))
            
            cv2.rectangle(self.image, top_left, bottom_right, rect_color, 3)

            for y in range(top_left[1], bottom_right[1]+1, self.line_spacing[i]):
                cv2.line(self.image, (top_left[0], y), (bottom_right[0], y), line_color, line_thickness)
            i=i+1

    def drawsave_results(self):
        image_path =self.imgname
        name_folderPath, ext = os.path.splitext(self.resname)
        txt_path = f"{name_folderPath}.txt"
        original_image = cv2.imread(image_path)


        for det in self.results.xyxy[0]:  
            xyxy = det[:4].cpu().numpy()  
            conf = det[4].cpu().item()  
            cls = det[5].cpu().item()  
            label = f'{self.results.names[int(cls)]} {conf:.2f}'  
            label0= self.results.names[int(cls)]

            cv2.rectangle(original_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,0,0), 2)
            cv2.putText(original_image, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        self.results_df.to_csv(txt_path, index=False, header=True, sep=' ')
        cv2.imwrite(self.resname, original_image)

    def pixel2length(self, area):
        ratio=self.dSB_Xlength.value()/self.imWidth
        scaled = [float(element * ratio) for element in area]
        return scaled
    
    def length2volt(self, area):
        ratio=1/self.dSB_L2volt.value()
        scaledV = [float(element * ratio) for element in area]
        return scaledV
    
    def refreshPorts(self):
        self.portCombo.clear()
        ports = list_ports.comports()
        for port in ports:
            self.portCombo.addItem(port.description, port.device)

    def do_Sercon(self):
        selectedPort = self.portCombo.currentData()
        try:
            if self.cB_Ethernet.isChecked():
                message=self.SPM_finder.connect_tcp_SPM(self.lE_Host.text(), self.lE_Port.text())
                self.displayFormattedText1(self.tBR_Serialput,message)
            else:
                message=self.SPM_finder.connect_to_SPM(selectedPort,9600)
                self.displayFormattedText1(self.tBR_Serialput,message)

        except Exception as e:
            self.displayFormattedText1(self.tBR_Serialput,f'Connection failed: {e}')

    def do_Serset(self):
        if self.cB_Serialout.isChecked():
            self.pB_SerConect.setEnabled(True)
            self.pB_SerDiscon.setEnabled(True)
        else:
            self.pB_SerConect.setEnabled(False)
            self.pB_SerDiscon.setEnabled(False)


    def do_Tcpip(self):
        if self.cB_Ethernet.isChecked():
            self.portCombo.setEnabled(False)
            self.lE_Port.setEnabled(True)
            self.lE_Host.setEnabled(True)
        else:
            self.portCombo.setEnabled(True)
            self.lE_Port.setEnabled(False)
            self.lE_Host.setEnabled(False)
    
    def do_Serdis(self):
            self.SPM_finder.client.close()
            self.SPM_finder.tcp_connection = True
            self.displayFormattedText1(self.tBR_Serialput,"Connection closed")

    def do_Editclass(self):
        file_clas=self.openFileNameDialog(5)
        if file_clas:
            subprocess.run(['notepad.exe',file_clas])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
