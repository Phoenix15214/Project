import socket
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from threading import Thread
import process_lib.control_lib as ctrl
import struct

message = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
server_socket = None
isConnected = False
pack = None


def _fatal_exit(reason, exc=None):
    if exc is not None:
        print(f"{reason}: {exc}")
    else:
        print(reason)
    # Called inside worker threads: force whole process to exit non-zero.
    os._exit(1)


def _init_pack():
    global pack
    if pack is not None:
        return pack
    try:
        pack = ctrl.SerialPacket(port="/dev/ttyUSB0", baudrate=38400, timeout=0.1)
    except Exception as exc:
        raise RuntimeError(f"无法打开串口: {exc}")
    return pack

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

def _send_thread(conns, method, socket):
    global pack
    global message
    # normalize to list
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while True:
        try:
            ready = wait(conns)
        except Exception:
            ready = []
        for r in ready:
            try:
                msg = r.recv()
            except Exception:
                continue
            if msg[0] == 0: # 来自track.py的消息
                message[0] = msg[1] # 偏移角度
                message[1] = msg[2] # 中心x值
                message[2] = msg[3] # 路口x值
                message[3] = msg[4] # 路口y值
                message[8] = msg[5] # 终点x值
                message[9] = msg[6] # 终点y值
                message[10] = msg[7] # 是否为丁字路口
            elif msg[0] == 1: # 来自detect.py的消息
                message[4] = msg[1] # 四个数字的ID
                message[5] = msg[2]
                message[6] = msg[3]
                message[7] = msg[4]
            if _init_pack() is not None:
                try:
                    pack.insert_byte(0x16)  # 包头
                    for i in range(11):
                        pack.insert_two_bytes(pack.num_to_bytes(message[i]))
                    pack.send_packet() # 发送数据包
                except Exception as exc:
                    _fatal_exit("串口发送失败，终止 transmit 进程", exc)
            try:
                if method == "firewater":
                    _send_by_firewater(message, socket)
                elif method == "justfloat":
                    _send_by_justfloat(message, socket)
            except Exception:
                print("客户端断开连接")
                # mark disconnected and notify producers if possible
                global isConnected
                isConnected = False
                for c in conns:
                    try:
                        c.send(isConnected)
                    except Exception:
                        pass
                return

def _recv_thread(conns, socket):
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while True:
        try:
            msg = socket.recv(1024).decode("utf8")
            if len(msg) == 0:
                break
            for c in conns:
                try:
                    c.send(msg)
                except Exception:
                    pass
        except Exception:
            break

def Listen_Thread(connect_socket):
    global server_socket
    global isConnected
    connect_socket.listen(3)
    server_socket, client_addr = connect_socket.accept()
    isConnected = True

def Empty_Thread(conns):
    global isConnected
    global message
    global pack
    if not isinstance(conns, (list, tuple)):
        conns = [conns]
    while not isConnected:
        try:
            ready = wait(conns, timeout=0.01)
        except Exception:
            ready = []
        for r in ready:
            try:
                msg = r.recv()
            except Exception:
                continue
            if msg[0] == 0:
                message[0] = msg[1]
                message[1] = msg[2]
                message[2] = msg[3]
                message[3] = msg[4]
                message[8] = msg[5]
                message[9] = msg[6]
                message[10] = msg[7]
            elif msg[0] == 1:
                message[4] = msg[1]
                message[5] = msg[2]
                message[6] = msg[3]
                message[7] = msg[4]
            if _init_pack() is not None:
                try:
                    pack.insert_byte(0x16)  # 包头
                    for i in range(11):
                        pack.insert_two_bytes(pack.num_to_bytes(message[i]))
                    pack.send_packet() # 发送数据包
                except Exception as exc:
                    _fatal_exit("串口发送失败，终止 transmit 进程", exc)


def Send_Process(conn, method="justfloat"):
    global server_socket
    global isConnected
    if method not in ("justfloat", "firewater"):
        print("发送方式不正确")
        method = "justfloat"
        print("自动更改格式为justfloat")
    isConnected = False
    _init_pack()
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connect_socket.bind(("", 11451))
    try:
        while True:    
            # normalize conn to list when starting helper threads
            conns = conn if isinstance(conn, (list, tuple)) else [conn]
            t0 = Thread(target=Listen_Thread, args=(connect_socket,))
            t00 = Thread(target=Empty_Thread, args=(conns,))
            t00.start()
            t0.start()
            # wait for client connection (Listen_Thread will set isConnected)
            t0.join()
            # ensure Empty_Thread has processed up to connection time
            t00.join()
            isConnected = True
            print("客户端已连接")
            t1 = Thread(target=_send_thread, args=(conns, method, server_socket))
            t2 = Thread(target=_recv_thread, args=(conns, server_socket))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            isConnected = False
            try:
                server_socket.close()
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
