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
from threading import Thread


eventlet.monkey_patch()

#数据集所在路径
DATA_DIR = "./data"
app.config['SECRET_KEY'] = 'your-secret-key'
os.makedirs(DATA_DIR, exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

app = Flask(__name__)

#临时存储WebSocketSID映射
active_sessions = {}

@app.route('/generate-quote', methods=['POST'])
def api_generate_quote():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON payload provided"}), 400

        data = json_data.get("data")
        method = json_data.get("method", "SHA-256")
        input_type = json_data.get("type", "string")

        if data is None:
            return jsonify({"error": "Missing required field: 'data'"}), 400

        allowed_types = {"string", "file_path", "binary"}
        if input_type not in allowed_types:
            return jsonify({"error": f"Invalid 'type'. Must be one of {allowed_types}"}), 400

        processed_data = None

        if input_type == "string":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a string when 'type' is 'string'"}), 400
            processed_data = data

        elif input_type == "file_path":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a string (file path) when 'type' is 'file_path'"}), 400
            if not os.path.exists(data):
                return jsonify({"error": f"File not found: {data}"}), 404
            if not os.path.isfile(data):
                return jsonify({"error": f"Not a valid file: {data}"}), 400
            processed_data = data 

        elif input_type == "binary":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a base64-encoded string when 'type' is 'binary'"}), 400
            try:
                binary_data = base64.b64decode(data, validate=True)
                processed_data = binary_data 
            except Exception as e:
                return jsonify({"error": f"Invalid base64 encoding in 'data': {str(e)}"}), 400

        result = generate_quote(data=processed_data, method=method, input_type=input_type)

        file_path = result.get("file_path")

        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=os.path.basename(file_path),
                mimetype='application/octet-stream'
            )
        else:
            return result["quote_binary"], 200, {
                'Content-Type': 'application/octet-stream',
                'Content-Disposition': f'attachment; filename="quote_{int(time.time())}.dat"'
            }

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/verify-quote', methods=['POST'])
def api_verify_quote():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON payload provided"}), 400

        data = json_data.get("data")
        input_type = json_data.get("type", "binary")  # 默认 binary

        allowed_types = {"file_path", "binary", "string"}
        if input_type not in allowed_types:
            return jsonify({"error": f"Invalid 'type'. Must be one of {allowed_types}"}), 400

        processed_data = None

        if input_type == "file_path":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a string (file path)"}), 400
            processed_data = data

        elif input_type == "binary":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a base64-encoded string when 'type' is 'binary'"}), 400
            try:
                binary_data = base64.b64decode(data, validate=True)
                processed_data = binary_data
            except Exception as e:
                return jsonify({"error": f"Invalid base64: {str(e)}"}), 400

        elif input_type == "string":
            if not isinstance(data, str):
                return jsonify({"error": "'data' must be a string"}), 400
            processed_data = data  # 传给 quote_verify

        result = quote_verify(data=processed_data, type=input_type)

        if result["success"]:
            return jsonify({
                "success": True,
                "payload": result["payload"]
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data-query', methods=['POST'])
def data_query():
    try:
        json_data = request.get_json()
        des_ip = json_data.get("des_ip")
        des_port = json_data.get("des_port")
        data_name = json_data.get("data_name")

        if not des_ip or not des_port or not data_name:
            return jsonify({"error": "Missing required fields: des_ip, des_port, data_name"}), 400

        if not isinstance(des_ip, str) or not isinstance(data_name, str):
            return jsonify({"error": "des_ip and data_name must be strings"}), 400

        try:
            des_port = int(des_port)
            if not (1 <= des_port <= 65535):
                raise ValueError
        except:
            return jsonify({"error": "des_port must be a valid integer port"}), 400

        target_url = f"http://{des_ip}:{des_port}/data-request"

        try:
            response = requests.post(
                target_url,
                json={"data_name": data_name},
                timeout=30
            )
        except requests.RequestException as e:
            return jsonify({
                "success": False,
                "error": f"Failed to connect to {des_ip}:{des_port}: {str(e)}"
            }), 500

        if response.status_code != 200:
            return jsonify({
                "success": False,
                "error": f"Remote error: {response.status_code} {response.text}"
            }), response.status_code

        remote_data = response.json()

        if not all(k in remote_data for k in ("data", "quote", "method")):
            return jsonify({"error": "Invalid response format from remote"}), 500

        data_b64 = remote_data["data"]        # base64 编码的数据
        quote_b64 = remote_data["quote"]      # base64 编码的 Quote
        method = remote_data["method"]        # 哈希算法

        try:
            data_bytes = base64.b64decode(data_b64)
            quote_bytes = base64.b64decode(quote_b64)
        except Exception as e:
            return jsonify({"error": f"Base64 decode failed: {str(e)}"}), 500

        fd, quote_file = tempfile.mkstemp(suffix=".dat")
        os.write(fd, quote_bytes)
        os.close(fd)

        verify_result = quote_verify(data=quote_file, type="file_path")

        if not verify_result["success"]:
            return jsonify({
                "success": False,
                "error": f"Quote verification failed: {verify_result['error']}"
            }), 400

        return jsonify({
            "success": True,
            "data_name": data_name,
            "data": data_b64, 
            "payload": verify_result["payload"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data-request', methods=['POST'])
def data_request():
    try:
        json_data = request.get_json()
        data_name = json_data.get("data_name")

        if not data_name:
            return jsonify({"error": "Missing 'data_name'"}), 400

        file_path = os.path.join(DATA_DIR, data_name)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return jsonify({"error": f"Data not found: {data_name}"}), 404

        with open(file_path, 'rb') as f:
            data_bytes = f.read()

        if len(data_bytes) == 0:
            return jsonify({"error": "Data is empty"}), 500

        method = "SHA-256"

        result = generate_quote(data=file_path, method=method, input_type="file_path")

        quote_bytes = result["quote_binary"]

        data_b64 = base64.b64encode(data_bytes).decode('ascii')
        quote_b64 = base64.b64encode(quote_bytes).decode('ascii')

        return jsonify({
            "data": data_b64,
            "quote": quote_b64,
            "method": method
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    active_sessions[request.sid] = None


@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    active_sessions.pop(request.sid, None)

@socketio.on('data_request_enhanced')
def handle_data_request(data):
    des_ip = data.get('des_ip')
    des_port = data.get('des_port')
    data_name = data.get('data_name')

    if not all([des_ip, des_port, data_name]):
        emit('error', {'msg': 'Missing required fields: des_ip, des_port, data_name'})
        return

    #启动后台任务连接远端
    active_sessions[request.sid] = {'status': 'connecting'}
    Thread(
        target=stream_from_remote,
        args=(request.sid, des_ip, des_port, data_name),
        daemon=True
    ).start()


def stream_from_remote(client_sid, des_ip, des_port, data_name):
    #连接远端data_provider，执行ECDH+Quote双向认证+AES-GCM流式接收
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((des_ip, des_port))
    except Exception as e:
        socketio.emit('error', {'msg': f'Connect failed: {e}'}, room=client_sid)
        return

    try:
        #1.发送本地公钥+quote
        priv_key, pub_pem = generate_ecdh_keypair()
        pub_b64 = base64.b64encode(pub_pem).decode()

        quote_result = generate_quote(data=pub_pem, method="SHA-256", input_type="binary")
        if not quote_result["success"]:
            socketio.emit('error', {'msg': f'Local quote gen failed: {quote_result["error"]}'}, room=client_sid)
            sock.close()
            return

        quote_b64 = base64.b64encode(quote_result["quote_binary"]).decode()

        msg = {
            "type": "pubkey",
            "pubkey": pub_b64,
            "quote": quote_b64,
            "data_name": data_name
        }
        sock.sendall((json.dumps(msg) + "\n").encode())

        #2.接收对方公钥+quote
        line = sock.recv(8192).decode().strip()
        resp = json.loads(line)
        peer_pub_pem = base64.b64decode(resp["pubkey"])
        peer_quote_b64 = resp["quote"]

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
            socketio.emit('error', {'msg': f'Quote file error: {e}'}, room=client_sid)
            sock.close()
            return

        if not verify_result["success"]:
            socketio.emit('error', {'msg': f'Quote verify failed: {verify_result["error"]}'}, room=client_sid)
            sock.close()
            return

        print("Peer Quote verified successfully")

        #3.计算共享密钥
        shared_key = derive_shared_key(priv_key, peer_pub_pem)
        aesgcm = AESGCM(shared_key)

        #4.接收START flag
        nonce = sock.recv(12)
        tag = sock.recv(16)
        ct = sock.recv(1024)
        try:
            plaintext = aesgcm.decrypt(nonce, ct + tag, None)
            if plaintext.decode().strip() != "START":
                raise ValueError("Invalid START")
        except Exception as e:
            socketio.emit('error', {'msg': f'Decrypt START failed: {e}'}, room=client_sid)
            sock.close()
            return

        print("Received START, begin receiving data...")

        #5.接收数据流
        buffer = b""
        while True:
            size_bytes = sock.recv(4)
            if len(size_bytes) == 0:
                break
            chunk_size = int.from_bytes(size_bytes, 'big')
            nonce = sock.recv(12)
            tag = sock.recv(16)
            ct = b""
            while len(ct) < chunk_size:
                ct += sock.recv(min(4096, chunk_size - len(ct)))
            try:
                chunk = aesgcm.decrypt(nonce, ct + tag, None)
            except Exception as e:
                socketio.emit('error', {'msg': f'Decrypt chunk failed: {e}'}, room=client_sid)
                break
            if chunk == b"EOF":
                break
            buffer += chunk

        print(f"Received total {len(buffer)} bytes")

        #6.解析并发送给前端
        try:
            data_json = json.loads(buffer.decode('utf-8'))
        except json.JSONDecodeError:
            data_json = {"raw_data_base64": base64.b64encode(buffer).decode()}

        socketio.emit('data', data_json, room=client_sid)

        #7.发送ack
        sock.sendall(b"ACK")

    except Exception as e:
        socketio.emit('error', {'msg': f'Stream error: {str(e)}'}, room=client_sid)
    finally:
        sock.close()
        active_sessions.pop(client_sid, None)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)