import socket
from multiprocessing import Process, Pipe
from threading import Thread
import process_lib.control_lib as ctrl
import struct
import time
import os

message = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# target_location = [0, 0, 0, 0] # 检测到目标的x坐标，未检测到时为0
server_socket = None
isConnected = False
pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)
last_send_time = 0

def Parse_Input(msg):
    start_flag = ":"
    end_flag = "\n"
    start_pos = msg.find(start_flag)
    content_pos = start_pos + len(start_flag)
    end_pos = msg.find(end_flag)
    if start_pos == -1 or end_pos == -1:
        return None, None
    if end_pos <= start_pos:
        return None, None
    command = msg[0:start_pos]
    value = msg[content_pos:end_pos]
    return command, value

def _send_by_firewater(data_list, socket):
    send_msg = ",".join(str(x) for x in data_list) + "\n"
    socket.send(send_msg.encode("utf8"))

def _send_by_justfloat(data_list, socket):
    format_string = '<' + 'f' * len(data_list)
    packed_data = struct.pack(format_string, *data_list)
    tail = b'\x00\x00\x80\x7f'
    socket.send(packed_data + tail)

def process_msg(message, msg):
    if msg[0] == 0: # 来自track.py的消息
            message[0] = msg[1] # 循迹偏离角度
            message[1] = msg[2] # 循迹线偏离中心x坐标
            message[2] = msg[3] # 路口x坐标
            message[3] = msg[4] # 路口y坐标
            message[8] = msg[5] # 终点x坐标
            message[9] = msg[6] # 终点y坐标
            message[10] = msg[7] # 是否垂直路口
    elif msg[0] == 1: # 来自detect.py的消息
            message[4] = msg[1] # 四个目标的类别ID
            message[5] = msg[2]
            message[6] = msg[3]
            message[7] = msg[4]

def _send_thread(conn, method, socket, stop_event):
    global pack
    global message
    while not stop_event.is_set():
        if not conn.poll(0.1):
            continue
        try:
            msg = conn.recv()
        except EOFError:
            break
        process_msg(message, msg)
        pack.insert_byte(0x16)  # 包头
        for i in range(11):
            pack.insert_two_bytes(pack.num_to_bytes(message[i]))
        pack.send_packet() # 发送数据包
        try:
            if method == "firewater":
                _send_by_firewater(message, socket)
            elif method == "justfloat":
                _send_by_justfloat(message, socket)
        except:
            print("客户端断开连接")
            isConnected = False
            try:
                conn.send(isConnected)
            except Exception:
                pass
            break

def _recv_thread(conn, sock, stop_event):
    sock.settimeout(0.2)
    while not stop_event.is_set():
        try:
            msg = sock.recv(1024).decode("utf8")
            if len(msg) == 0:
                break
            conn.send(msg)
        except socket.timeout:
            continue
        except:
            break

def Listen_Thread(connect_socket, stop_event):
    global server_socket
    global isConnected
    connect_socket.listen(3)
    connect_socket.settimeout(0.2)
    while not stop_event.is_set():
        try:
            server_socket, client_addr = connect_socket.accept()
            isConnected = True
            return
        except socket.timeout:
            continue
        except Exception:
            return

def Empty_Thread(conn, stop_event):
    global isConnected
    global message
    global pack
    global last_send_time
    while not isConnected and not stop_event.is_set():
        if conn.poll(0.01):
            try:
                msg = conn.recv()
            except EOFError:
                break
            process_msg(message, msg)
            pack.insert_byte(0x16)  # 包头
            for i in range(11):
                pack.insert_two_bytes(pack.num_to_bytes(message[i]))
            pack.send_packet() # 发送数据包


def Send_Process(conn, method="justfloat", stop_event=None, core=None):
    if core is not None:
        os.sched_setaffinity(0, {core})
    global server_socket
    global isConnected
    stop_event = stop_event or type("StopEvent", (), {"is_set": staticmethod(lambda: False)})()
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    isConnected = False
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connect_socket.bind(("", 11451))
    connect_socket.listen(3)
    try:
        while not stop_event.is_set():
            t0 = Thread(target=Listen_Thread, args=(connect_socket, stop_event), daemon=True)
            t00 = Thread(target=Empty_Thread, args=(conn, stop_event), daemon=True)
            t00.start()
            t0.start()
            while t0.is_alive() and not stop_event.is_set():
                t0.join(timeout=0.2)
            while t00.is_alive() and not stop_event.is_set():
                t00.join(timeout=0.2)
            if stop_event.is_set():
                break
            isConnected = True
            print("客户端已连接")
            t1 = Thread(target=_send_thread, args=(conn, method, server_socket, stop_event), daemon=True)
            t2 = Thread(target=_recv_thread, args=(conn, server_socket, stop_event), daemon=True)
            t1.start()
            t2.start()
            while (t1.is_alive() or t2.is_alive()) and not stop_event.is_set():
                t1.join(timeout=0.2)
                t2.join(timeout=0.2)
            isConnected = False
            try:
                server_socket.close()
            except Exception:
                pass
        try:
            connect_socket.close()
        except Exception:
            pass
    finally:
        try:
            connect_socket.close()
            server_socket.close()
        except Exception:
            pass
    
if __name__ == "__main__":
    parent_conn, child_conn = Pipe()
    p_send = Process(target=Send_Process, args=(child_conn, "justfloat"))
    p_send.start()
    while True:
        msg = parent_conn.recv()
        print("接收到数据:", msg)