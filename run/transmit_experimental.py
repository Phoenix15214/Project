import socket
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from threading import Thread
import process_lib.control_lib as ctrl
import struct

# 定义全局变量
message = []
pack = None
server_socket = None
isConnected = False
send_ready = [False, False]  # 用于标记两个发送线程是否准备好发送数据,前为网络线程，后为串口线程
config = ctrl.ConfigManager("config.json")
config_data = config.get_all()

def init_message(length):
    global message
    message = [0] * length

# 遇到报错，退出程序并留下非零的退出码
def _fatal_exit(reason, exc=None):
    if exc is not None:
        print(f"{reason}: {exc}")
    else:
        print(reason)
    # Called inside worker threads: force whole process to exit non-zero.
    os._exit(1)

# 初始化串口
def _init_pack(port="/dev/ttyUSB0", baudrate=115200):
    global pack
    if pack is not None:
        return pack
    try:
        pack = ctrl.SerialPacket(port=port, baudrate=baudrate, timeout=0.1)
    except Exception as exc:
        raise RuntimeError(f"无法打开串口: {exc}")
    return pack

def Send_Thread_Network(method, socket):
    global pack
    global message
    global send_ready
    while True:
        if not send_ready[0]:
            time.sleep(0.01)
            continue
        try:
            if method == "justfloat":
                ctrl._send_by_justfloat(message, socket)
            elif method == "firewater":
                ctrl._send_by_firewater(message, socket)
        except Exception as exc:
            print(f"网络发送失败: {exc}")
            send_ready[0] = False
            continue

# 监听并创建socket连接
def Listen_Thread(connect_socket):
    global server_socket
    global isConnected
    connect_socket.listen(3)
    server_socket, client_addr = connect_socket.accept()
    isConnected = True

# 始终存在，更新message并防止pipe阻塞
def Update_Thread(conn):
    global message
    global send_ready
    if conn is None:
        return
    while True:
        if not conn.poll(0.1):
            continue
        try:
            msg = conn.recv()
            send_ready = [True, True]
        except EOFError:
            break
        message = msg

# 网络传输线程，调度两个子线程分别进行收发
def Network_Thread(conn, socket, method="justfloat"):
    global server_socket
    global isConnected
    isConnected = False
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    
    _init_pack()
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connect_socket.bind(("", 11451))

