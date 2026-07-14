import socket
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from threading import Thread
import process_lib.control_lib as ctrl
import struct
import asyncio

message = []
pack = None
server_socket = None
config = ctrl.ConfigManager("config.json")
config_data = config.get_all()

def _fatal_exit(reason, exc=None):
    if exc is not None:
        print(f"{reason}: {exc}")
    else:
        print(reason)
    os._exit(1)

def _init_pack(port="/dev/ttyUSB0", baudrate=115200):
    global pack
    if pack is not None:
        return pack
    try:
        pack = ctrl.SerialPacket(port=port, baudrate=baudrate, timeout=0.1)
    except Exception as exc:
        raise RuntimeError(f"无法打开串口: {exc}")
    return pack

def init_message(length):
    global message
    message = [0] * length

# 更新发送内容
def update_message(new_message):
    global message
    message = new_message

# 更新配置信息
async def Update_Config(require_refresh: asyncio.Event):
    global config
    global config_data
    while True:
        await require_refresh.wait()
        config.update()
        config_data = config.get_all()
        require_refresh.clear()

# 监听并创建socket连接
async def Listen_Accept(connect_socket, Connected: asyncio.Event):
    global server_socket
    while True:
        try:
            connect_socket.listen(3)
            server_socket, addr = connect_socket.accept()
            print(f"Accepted connection from {addr}")
            Connected.set()
        except Exception as exc:
            print(f"Error accepting connection: {exc}")

# 获取管道中的消息并更新全局message变量
async def Aquire_Message(conn, send_ready: asyncio.Event):
    while conn is not None:
        try:
            # 等待管道中有数据可读
            msg = await asyncio.to_thread(conn.recv)
            update_message(msg)
            send_ready.set()  # 设置事件，表示有新消息可发送
        except EOFError:
            break

async def Send(method, pack, send_ready:asyncio.Event):
    global message
    global server_socket
    while True:
        try:
            await send_ready.wait()
            print(server_socket)
            if server_socket is not None:
                if method == "justfloat":
                    ctrl._send_by_justfloat(message, server_socket)
                elif method == "firewater":
                    ctrl._send_by_firewater(message, server_socket)
            if pack is not None:
                pack.insert_byte(len(message))  # 包头
                for i in range(len(message)):
                    pack.insert_three_bytes(pack.num_to_bytes(message[i]))
                pack.send_packet()
            send_ready.clear()
        except Exception as exc:
            print(f"发送失败: {exc}")
            server_socket = None

# 从socket接收数据并解析配置更新
async def Recv_Network(socket, require_refresh: asyncio.Event, Connected: asyncio.Event):
    global config
    global config_data
    while True:
        await Connected.wait()  # 等待连接建立
        while socket is not None:
            try:
                msg = await asyncio.get_event_loop().sock_recv(socket, 1024)
                if len(msg) == 0:
                    break
                command, value = pack.parse_input(msg)
                if command == "start":
                    config.update()
                else:
                    original_value = config_data.get(command, None)
                    if original_value is not None:
                        config.set_value(command, int(value))
                        config.save()
                        require_refresh.set()  # 设置事件，表示配置已更新
            except Exception:
                break
        Connected.clear()  # 连接断开，清除事件
    Connected.clear()  # 确保在退出前清除事件

async def Recv_Serial(pack, require_refresh: asyncio.Event):
    global config
    global config_data
    while pack is not None:
        try:
            msg = await asyncio.to_thread(pack.receive_packet, 0.02)
            if msg is None:
                continue
            command, value = pack.parse_input(msg)
            if command == "start":
                config.update()
            else:
                original_value = config_data.get(command, None)
                if original_value is not None:
                    config.set_value(command, int(value))
                    config.save()
                    require_refresh.set()  # 设置事件，表示配置已更新
        except Exception:
            break

async def Tik_Tok(send_ready: asyncio.Event, interval: float):
    while True:
        await asyncio.sleep(interval)
        send_ready.set()
        print("Tik_Tok: send_ready set")

async def main(conn, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"):
    global pack
    global server_socket
    global message
    global config_data
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    connect_socket.bind(("", 11451))

    pack = _init_pack(port, baudrate)
    init_message(len(config_data))
    require_refresh = asyncio.Event()
    send_ready = asyncio.Event()
    Connected = asyncio.Event()
    require_refresh.set()

    # 创建任务
    tasks = [
        asyncio.create_task(Aquire_Message(conn, send_ready)), # 从管道获取消息
        asyncio.create_task(Send(method, pack, send_ready)), # 发送消息到网络和串口
        asyncio.create_task(Recv_Network(server_socket, require_refresh, Connected)), # 从网络接收配置更新
        asyncio.create_task(Recv_Serial(pack, require_refresh)), # 从串口接收配置更新
        asyncio.create_task(Update_Config(require_refresh)), # 更新配置
        asyncio.create_task(Listen_Accept(connect_socket, Connected)), # 监听并接受网络连接
        asyncio.create_task(Tik_Tok(send_ready, 0.05)), # 定时触发发送
    ]

    # 等待所有任务完成
    await asyncio.gather(*tasks)

asyncio.run(main(None, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"))
