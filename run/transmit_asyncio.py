import socket
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait
from threading import Thread
import process_lib.control_lib as ctrl
import struct
import asyncio
import time

config_message = []
message = [0, 0, 0, 0, 0, 0, 0, 0]
pack = None
server_socket = None
config = ctrl.ConfigManager("config.json")
config_data = config.get_all()
command_to_send = ""
send_command_ready = False
start_send_time = time.time()

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
        # raise RuntimeError(f"无法打开串口: {exc}")
    return pack

def init_message(length):
    global message
    message = [0] * length
    global config_message
    config_message = [0] * length

# 更新发送内容
def update_config_message():
    global config_message
    global config_data
    new_message = []
    for value in config_data.values():
        new_message.append(value)
    config_message = new_message

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
        update_config_message()
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
    global command_to_send
    global send_command_ready
    global start_send_time
    while conn is not None:
        try:
            # 等待管道中有数据可读
            msg = await asyncio.to_thread(conn.recv)
            new_message = [0, 0, 0, 0, 0, 0, 0, 0]
            if msg[0] == 0: # 0开头为正常数据更新
                for i in range(8):
                    new_message[i] = msg[i + 1]
                update_message_manual(new_message)
                send_ready_network.set()  # 设置事件，表示有新消息可发送
                send_ready_serial.set()  # 设置事件，表示有新消息可发送
            elif msg[0] == 1: # 1开头为文本消息,需要反复发送
                command_to_send = msg[1]
                send_command_ready = True
                start_send_time = time.time()
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

async def Send_Serial(pack, send_ready: asyncio.Event):
    global message
    global command_to_send
    global send_command_ready
    global start_send_time
    while True:
        try:
            await send_ready.wait()
            if pack is not None:
                pack.insert_byte(0x09)  # 包头
                pack.insert_three_bytes(pack.num_to_bytes(0))
                for i in range(len(message)):
                    pack.insert_three_bytes(pack.num_to_bytes(message[i]))
                pack.send_packet()
            if send_command_ready and pack is not None:
                pack.send_char(command_to_send)
                if time.time() - start_send_time > 1: # 一秒超时
                    send_command_ready = False
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
                command, value = pack.parse_input(msg)
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

async def Recv_Serial(conn, pack, require_refresh: asyncio.Event):
    global config
    global config_data
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
                pack.insert_byte(0x07) # 数据量，六个阈值
                pack.insert_three_bytes(pack.num_to_bytes(1))
                for val in config_data.values():
                    pack.insert_three_bytes(pack.num_to_bytes(int(val)))
                    print(f"Sending config value: {val}")
                for i in range (5):
                    pack.send_packet()
                    await asyncio.sleep(0.05)
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

async def Tik_Tok(send_ready_network: asyncio.Event, send_ready_serial: asyncio.Event, interval: float):
    while True:
        await asyncio.sleep(interval)
        send_ready_network.set()
        send_ready_serial.set()

async def _wait_stop_event(stop_event):
    if stop_event is None:
        return
    await asyncio.to_thread(stop_event.wait)


async def _run_workers(tasks):
    await asyncio.gather(*tasks)


async def main_task(conn, stop_event=None, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"):
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
    print(f"程序已启动，监听端口: 11451, 串口: {port}, 波特率: {baudrate}, 发送方法: {method}")

    # 创建任务
    tasks = [
        asyncio.create_task(Aquire_Message(conn, send_ready_network, send_ready_serial)), # 从管道获取消息
        asyncio.create_task(Send_Network(method, send_ready_network)), # 发送消息到网络
        asyncio.create_task(Send_Serial(pack, send_ready_serial)), # 发送消息到串口
        asyncio.create_task(Recv_Network(require_refresh, Connected)), # 从网络接收配置更新
        asyncio.create_task(Recv_Serial(conn, pack, require_refresh)), # 从串口接收配置更新
        asyncio.create_task(Update_Config(require_refresh)), # 更新配置
        asyncio.create_task(Listen_Accept(connect_socket, Connected)), # 监听并接受网络连接
        asyncio.create_task(Tik_Tok(send_ready_network, send_ready_serial, 0.05)), # 定时触发发送
    ]

    try:
        if stop_event is None:
            await asyncio.gather(*tasks)
        else:
            stop_wait_task = asyncio.create_task(_wait_stop_event(stop_event))
            workers_task = asyncio.create_task(_run_workers(tasks))
            done, pending = await asyncio.wait(
                {stop_wait_task, workers_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if workers_task in done:
                await workers_task
            else:
                stop_event.set()

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if server_socket is not None:
            try:
                server_socket.close()
            except Exception:
                pass
            server_socket = None
        connect_socket.close()

def main(conn=None, stop_event=None, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"):
    try:
        asyncio.run(main_task(conn, stop_event, port, baudrate, method))
    except KeyboardInterrupt:
        if stop_event is not None:
            stop_event.set()
        raise
    except Exception:
        if stop_event is not None:
            stop_event.set()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main_task(None, port="/dev/ttyUSB0", baudrate=115200, method="justfloat"))
    except KeyboardInterrupt:
        print("程序已终止")
    except Exception as exc:
        _fatal_exit("发生错误", exc)
