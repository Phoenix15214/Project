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
        print("无法打开串口")
        pack = None
    return pack

def init_message(length):
    global message
    message = [0] * length

# 更新发送内容
def update_message():
    global message
    global config_data
    new_message = []
    for value in config_data.values():
        new_message.append(value)
    message = new_message

def update_message_manual(new_message):
    global message
    message = new_message

# 更新配置信息
async def Update_Config(require_refresh: asyncio.Event):
    global config
    global config_data
    while True:
        await require_refresh.wait()
        print("配置已更新，当前配置为:", config_data)
        config.update()
        config_data = config.get_all()
        # update_message()
        require_refresh.clear()

# 监听并创建socket连接
async def Listen_Accept(connect_socket, Connected: asyncio.Event):
    global server_socket
    loop = asyncio.get_event_loop()
    connect_socket.setblocking(False)
    while True:
        try:
            connect_socket.listen(3)
            server_socket, addr = await loop.sock_accept(connect_socket)
            print(f"Accepted connection from {addr}")
            Connected.set()
        except Exception as exc:
            print(f"Error accepting connection: {exc}")

# 获取管道中的消息并更新全局message变量
async def Aquire_Message(conn, send_ready_network: asyncio.Event, send_ready_serial: asyncio.Event):
    global message
    global pack
    while conn is not None:
        try:
            # 等待管道中有数据可读
            msg = await asyncio.to_thread(conn.recv)
            new_message = [0, 0]
            if msg[0] == 0:
                new_message[0] = msg[1]
                new_message[1] = msg[2]
                update_message_manual(new_message)
            elif msg[0] == 1:
                send_message = msg[1]
                if pack is not None:
                    pack.send_char(send_message)
            send_ready_network.set()  # 设置事件，表示有新消息可发送
            send_ready_serial.set()  # 设置事件，表示有新消息可发送
        except EOFError:
            break

async def Send_Network(method, send_ready:asyncio.Event):
    global message
    global server_socket
    while True:
        try:
            await send_ready.wait()
            if server_socket is not None:
                if method == "justfloat":
                    ctrl._send_by_justfloat(message, server_socket)
                elif method == "firewater":
                    ctrl._send_by_firewater(message, server_socket)
        except Exception as exc:
            print(f"发送失败: {exc}")
            server_socket = None
            continue
        finally:
            send_ready.clear()
    send_ready.clear()

async def Send_Serial(send_ready: asyncio.Event):
    global pack
    global message
    while True:
        try:
            await send_ready.wait()
            if pack is not None:
                pack.insert_byte(0x03)  # 包头
                pack.insert_three_bytes(pack.num_to_bytes(0))
                for i in range(len(message)):
                    pack.insert_three_bytes(pack.num_to_bytes(message[i]))
                pack.send_packet()
            send_ready.clear()
        except Exception as exc:
            print(f"串口发送失败: {exc}")
            continue
        finally:
            send_ready.clear()
    send_ready.clear()  # 确保在退出前清除事件

# 从socket接收数据并解析配置更新
async def Recv_Network(require_refresh: asyncio.Event, Connected: asyncio.Event):
    global config
    global config_data
    global server_socket
    while True:
        await Connected.wait()  # 等待连接建立
        while server_socket is not None:
            try:
                msg = await asyncio.get_event_loop().sock_recv(server_socket, 1024)
                msg = msg.decode('utf-8').strip()
                print(f"Received data: {msg}")
                if len(msg) == 0:
                    break
                command, value = ctrl.Parse_Input(msg)
                if command == "start":
                    config.update()
                elif command is None:
                    print(f"无法解析的命令: {msg}")
                else:
                    original_value = config_data.get(command, None)
                    if original_value is not None:
                        config.set_value(command, int(value))
                        config.save()
                        require_refresh.set()  # 设置事件，表示配置已更新
            except Exception as e:
                print(f"网络接收失败: {e}")
                break
        Connected.clear()
    Connected.clear()

async def Recv_Serial(conn, require_refresh: asyncio.Event):
    global config
    global config_data
    global pack
    while pack is not None:
        try:
            msg_ready = await asyncio.to_thread(pack.recv_packet, 0.02)
            if not msg_ready:
                continue
            msg = pack.get_recv_data()
            print(f"Received serial data: {msg}")
            command, value = pack.parse_input(msg)
            if command == "start":
                config.update()
                pack.insert_byte(0x06)
                pack.insert_three_bytes(pack.num_to_bytes(1))
                for val in config_data.values():
                    pack.insert_three_bytes(pack.num_to_bytes(int(val)))
                pack.send_packet()
            elif command == "Con_Mode_1":
                print("收到题目一指令")
                if conn is not None:
                    conn.send("task:1\n")
            elif command == "Con_Mode_2":
                if conn is not None:
                    conn.send("task:2\n")
            elif command == "Con_Mode_3":
                if conn is not None:
                    conn.send("task:3\n")
            elif command == "Con_Mode_4":
                if conn is not None:
                    conn.send("task:4\n")
            elif command == "Con_Mode_5":
                if conn is not None:
                    conn.send("task:5\n")
            elif command == "Move":
                if conn is not None:
                    conn.send(f"Move:{value}\n")
            elif command == "OK":
                if value == "4":
                    send_message = "@Down$#"
                    pack.send_char(send_message)
                if conn is not None:
                    conn.send(f"OK:{value}\n")
            else:
                original_value = config_data.get(command, None)
                if original_value is not None:
                    config.set_value(command, int(value))
                    config.save()
                    send_message = "@Get$#"
                    pack.send_char(send_message)
                    require_refresh.set()  # 设置事件，表示配置已更新
        except Exception as e:
            print(f"串口接收失败: {e}")
            break

async def Tik_Tok(send_ready_network: asyncio.Event, send_ready_serial: asyncio.Event, interval: float):
    while True:
        await asyncio.sleep(interval)
        send_ready_network.set()
        send_ready_serial.set()

async def main_task(conn, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"):
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
    send_ready_network = asyncio.Event()
    send_ready_serial = asyncio.Event()
    Connected = asyncio.Event()
    require_refresh.set()

    # 创建任务
    tasks = [
        asyncio.create_task(Aquire_Message(conn, send_ready_network, send_ready_serial)), # 从管道获取消息
        asyncio.create_task(Send_Network(method, send_ready_network)), # 发送消息到网络
        asyncio.create_task(Send_Serial(send_ready_serial)), # 发送消息到串口
        asyncio.create_task(Recv_Network(require_refresh, Connected)), # 从网络接收配置更新
        asyncio.create_task(Recv_Serial(conn, require_refresh)), # 从串口接收配置更新
        asyncio.create_task(Update_Config(require_refresh)), # 更新配置
        asyncio.create_task(Listen_Accept(connect_socket, Connected)), # 监听并接受网络连接
        asyncio.create_task(Tik_Tok(send_ready_network, send_ready_serial, 0.05)), # 定时触发发送
    ]

    # 等待所有任务完成
    await asyncio.gather(*tasks)

def main(conn=None, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"):
    asyncio.run(main_task(conn, port, baudrate, method))


if __name__ == "__main__":
    main()
