import torch,cv2,random,os,time
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import argparse
import threading, queue
from torch.multiprocessing import Pool, Process, set_start_method
from darknet import Darknet
from imutils.video import WebcamVideoStream,FPS

import win32com.client as wincl
spk = wincl.Dispatch("SAPI.SpVoice")

torch.multiprocessing.set_start_method('spawn', force=True)


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def lttbox_img(img, inp_dim):

    image_width, image_height = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_width = int(image_width * min(w/image_width, h/image_height))
    new_height = int(image_height * min(w/image_width, h/image_height))
    res_img = cv2.resize(img, (new_width,new_height), interpolation = cv2.INTER_CUBIC)
    cnvs = np.full((inp_dim[1], inp_dim[0], 3), 128)
    cnvs[(h-new_height)//2:(h-new_height)//2 + new_height,(w-new_width)//2:(w-new_width)//2 + new_widthidth,  :] = res_img

    return cnvs

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def prepare_img(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    original_image = img
    dim = original_image.shape[1], original_image.shape[0]
    img = (lttbox_img(original_image, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, original_image, dim



class ObjectDetection:
    def __init__(self, id):
        # self.cap = cv2.VideoCapture(id)
        self.cap = WebcamVideoStream(src = id).start()
        self.cfgfile = "cfg/yolov4.cfg"
        self.weightsfile = "yolov4.weights"
        self.confidence = float(0.6)
        self.nms_thesh = float(0.8)
        self.num_classes = 80
        self.classes = load_classes('data/coco.names')
        self.colors = pkl.load(open("pallete", "rb"))
        self.model = Darknet(self.cfgfile)
        self.CUDA = torch.cuda.is_available()
        self.model.load_weights(self.weightsfile)
        self.width = 1280 #640#1280
        self.height = 720 #360#720
        
        if self.CUDA:
            self.model.cuda()


        self.model.eval()

    def main(self):
        q = queue.Queue()
        while True:
            def frame_render(queue_from_cam):
                frame = self.cap.read() # If you capture stream using opencv (cv2.VideoCapture()) the use the following line
                # ret, frame = self.cap.read()
                frame = cv2.resize(frame,(self.width, self.height))
                queue_from_cam.put(frame)
            cam = threading.Thread(target=frame_render, args=(q,))
            cam.start()
            cam.join()
            frame = q.get()
            q.task_done()
            fps = FPS().start()

            try:
                img, original_image, dim = prepare_img(frame, 160)

                im_dim = torch.FloatTensor(dim).repeat(1,2)
                if self.CUDA:                            #### If you have a gpu properly installed then it will run on the gpu
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                # with torch.no_grad():               #### Set the model in the evaluation mode

                output = self.model(img)
                from tool.utils import post_processing,plot_boxes_cv2
                bounding_boxes = post_processing(img,self.confidence, self.nms_thesh, output)
                frame = plot_boxes_cv2(frame, bounding_boxes[0], savename= None, class_names=self.classes, color = None, colors=self.colors)

            except:
                pass

            fps.update()
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))

            cv2.imshow("Object Detection Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            torch.cuda.empty_cache()



if __name__ == "__main__":
    id = 0
    ObjectDetection(id).main()
