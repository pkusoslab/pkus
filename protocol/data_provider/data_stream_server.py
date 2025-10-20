from flask import Flask, request, jsonify, send_file
#pip install flask-socketio eventlet，支持管道
from flask_socketio import SocketIO, emit, disconnect
from shared.quote_generation import generate_quote
from shared.quote_verification import verify_quote
from shared.encrypt_utils import generate_ecdh_keypair, derive_shared_key
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64
import time
import json
import socket
import eventlet
import threading
from pathlib import Path


HOST = '0.0.0.0'
PORT = 9000
DATA_DIR = Path("./data")

def handle_client(sock):
    try:
        #1.接收请求方公钥+quote
        line = sock.recv(8192).decode().strip()
        msg = json.loads(line)
        peer_pub_pem = base64.b64decode(msg["pubkey"])
        peer_quote_b64 = msg["quote"]
        data_name = msg["data_name"]

        data_path = DATA_DIR / data_name
        if not data_path.exists():
            print(f"Data not found: {data_path}")
            sock.close()
            return

        #验证对方quote
        fd, quote_file = None, None
        try:
            fd, quote_file = os.mkstemp(suffix=".quote")
            os.write(fd, base64.b64decode(peer_quote_b64))
            os.close(fd)
            verify_result = quote_verify(data=quote_file, type="file_path")
            os.unlink(quote_file)
        except Exception as e:
            if fd:
                os.close(fd)
            if quote_file and os.path.exists(quote_file):
                os.unlink(quote_file)
            sock.close()
            return

        if not verify_result["success"]:
            print(f"Quote verify failed: {verify_result['error']}")
            sock.close()
            return

        print("Requester's Quote verified")

        #2.发送自己的公钥+quote
        priv_key, pub_pem = generate_ecdh_keypair()
        pub_b64 = base64.b64encode(pub_pem).decode()

        quote_result = generate_quote(data=pub_pem, method="SHA-256", input_type="binary")
        if not quote_result["success"]:
            sock.close()
            return

        quote_b64 = base64.b64encode(quote_result["quote_binary"]).decode()

        response = {
            "type": "pubkey",
            "pubkey": pub_b64,
            "quote": quote_b64
        }
        sock.sendall((json.dumps(response) + "\n").encode())

        #3.计算共享密钥
        shared_key = derive_shared_key(priv_key, peer_pub_pem)
        aesgcm = AESGCM(shared_key)

        #4.发送 START flag
        nonce = os.urandom(12)
        ct, tag = aesgcm.encrypt(nonce, b"START", None)
        sock.sendall(nonce + tag + ct)

        #5.流式发送数据
        with open(data_path, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                nonce = os.urandom(12)
                ct, tag = aesgcm.encrypt(nonce, chunk, None)
                sock.sendall(len(ct).to_bytes(4, 'big') + nonce + tag + ct)

        #6.发送eof
        nonce = os.urandom(12)
        ct, tag = aesgcm.encrypt(nonce, b"EOF", None)
        sock.sendall(nonce + tag + ct)

        #7.等待ack
        try:
            ack = sock.recv(3)
            if ack == b"ACK":
                print("ACK received, transfer complete")
        except:
            pass

    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        sock.close()


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"✅ Data Provider Server listening on {HOST}:{PORT}")

    while True:
        client_sock, addr = server.accept()
        print(f"🔗 Incoming connection from {addr}")
        threading.Thread(target=handle_client, args=(client_sock,), daemon=True).start()


if __name__ == '__main__':
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
        #创建测试数据
        test_data = {"message": "Hello from TEE!", "timestamp": __import__('time').time()}
        #假定测试数据在./data/test_data.json
        (DATA_DIR / "test_data.json").write_text(json.dumps(test_data, indent=2))
        print("Created test data file")

    start_server()