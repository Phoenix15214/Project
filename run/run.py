import os
import cv2
import numpy as np
import process_lib.control_lib as ctrl
from multiprocessing import Process, Pipe, shared_memory, Value, Event
from threading import Thread
import time
from transmit import Send_Process

def main():
    


