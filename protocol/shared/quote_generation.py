import hashlib
import subprocess
import time
import os
from typing import Union, BinaryIO
import blake3

TDX_ATTEST_TOOL = "/home/tdx/tdx_attest/app"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_quote(data: Union[str, bytes, BinaryIO], method: str, input_type: str = None) -> dict:

    supported_methods = {"SHA-256", "SHA-384", "SHA-512", "SHA3-256", "SHA3-384", "BLAKE3"}
    if method not in supported_methods:
        raise ValueError(f"Unsupported method: {method}")

    data_bytes = None

    if input_type == "string":
        if not isinstance(data, str):
            raise TypeError("For type 'string', data must be a string.")
        data_bytes = data.encode('utf-8')

    elif input_type == "file_path":
        if not isinstance(data, str):
            raise TypeError("For type 'file_path', data must be a string (path).")
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        if not os.path.isfile(data):
            raise ValueError(f"Not a file: {data}")
        with open(data, 'rb') as f:
            data_bytes = f.read()

    elif input_type == "binary":
        if not isinstance(data, bytes):
            raise TypeError("For type 'binary_bytes', data must be bytes.")
        data_bytes = data

    ## 仅供测试用，针对全内存对象，尚未完全实现
    elif input_type == "file_object":
        if not (hasattr(data, 'read') and callable(data.read)):
            raise TypeError("For type 'file_object', data must be a readable file-like object.")
        current_pos = None
        try:
            current_pos = data.tell()
        except (AttributeError, OSError):
            pass
        data.seek(0)
        data_bytes = data.read()
        if current_pos is not None:
            try:
                data.seek(current_pos)
            except:
                pass

    else:
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            if os.path.exists(data) and os.path.isfile(data):
                with open(data, 'rb') as f:
                    data_bytes = f.read()
            else:
                data_bytes = data.encode('utf-8')
        elif hasattr(data, 'read') and callable(data.read):
            current_pos = None
            try:
                current_pos = data.tell()
            except (AttributeError, OSError):
                pass
            data.seek(0)
            data_bytes = data.read()
            if current_pos is not None:
                try:
                    data.seek(current_pos)
                except:
                    pass
        else:
            raise TypeError("Unsupported data type.")

    if len(data_bytes) == 0:
        raise ValueError("Input data is empty.")

    if not os.path.exists(TDX_ATTEST_TOOL):
        raise FileNotFoundError(f"TDX attest tool not found: {TDX_ATTEST_TOOL}")

    if method == "SHA-256":
        hash_hex = hashlib.sha256(data_bytes).hexdigest()
    elif method == "SHA-384":
        hash_hex = hashlib.sha384(data_bytes).hexdigest()
    elif method == "SHA-512":
        hash_hex = hashlib.sha512(data_bytes).hexdigest()
    elif method == "SHA3-256":
        hash_hex = hashlib.sha3_256(data_bytes).hexdigest()
    elif method == "SHA3-384":
        hash_hex = hashlib.sha3_384(data_bytes).hexdigest()
    elif method == "BLAKE3":
        hash_hex = blake3.blake3(data_bytes).hexdigest()

    cmd = ["sudo", TDX_ATTEST_TOOL, "-d", hash_hex]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        quote_binary = result.stdout
        if len(quote_binary) == 0:
            raise RuntimeError("TDX attest tool returned empty output.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
        stdout = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ''
        raise RuntimeError(f"TDX attest failed: {stderr or stdout or 'Unknown error'}")
    except Exception as e:
        raise RuntimeError(f"Execution failed: {str(e)}")

    current_time = int(time.time())
    filename = f"quote_{current_time}.dat"
    file_path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(file_path, 'wb') as f:
            f.write(quote_binary)
        print(f"[INFO] Quote saved to {file_path} (size: {len(quote_binary)} bytes)")
    except Exception as e:
        raise RuntimeError(f"Failed to save .dat file: {str(e)}")

    return {
        "file_path": file_path,
        "quote_binary": quote_binary,
        "method_used": method,
        "timestamp": current_time,
        #"input_hash": hash_hex
    }