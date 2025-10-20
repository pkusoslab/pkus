# quote_verify.py
import os
import subprocess
import tempfile
import json
from typing import Union, Dict, Any

VERIFIER_TOOL = "/home/tdx/tdx_attest/verifier"
RELYING_PARTY_TOOL = "/home/tdx/tdx_attest/relying_party"
TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)


def verify_quote(data: Union[str, bytes], type: str) -> Dict[str, Any]:

    quote_file_path = None

    try:
        quote_binary = None

        if type == "file_path":
            if not isinstance(data, str):
                raise TypeError("For type 'file_path', data must be a string (path).")
            if not os.path.exists(data):
                raise FileNotFoundError(f"Quote file not found: {data}")
            if not os.path.isfile(data):
                raise ValueError(f"Not a file: {data}")
            quote_file_path = data

        elif type == "binary":
            if not isinstance(data, bytes):
                raise TypeError("For type 'binary', data must be bytes.")
            if len(data) == 0:
                raise ValueError("Empty binary data.")
            quote_binary = data

        elif type == "string":
            if not isinstance(data, str):
                raise TypeError("For type 'string', data must be a string.")
            try:
                quote_binary = base64.b64decode(data, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid base64 string: {str(e)}")
        else:
            raise ValueError(f"Unsupported type: {type}")

        #对于非文件路径，先存储到临时文件，用于验证
        if quote_binary is not None:
            fd, quote_file_path = tempfile.mkstemp(suffix=".dat", dir=TEMP_DIR)
            try:
                os.write(fd, quote_binary)
            finally:
                os.close(fd)

        cmd = (
            f"sudo {VERIFIER_TOOL} -d {quote_file_path} | "
            f"{RELYING_PARTY_TOOL} -v | "
            f"grep 'json payload' | "
            f"awk -F 'payload:' '{{print $2}}' | "
            f"jq"
        )

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Verification command timed out",
                "payload": None,
                "raw_output": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "payload": None,
                "raw_output": ""
            }

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            error_msg = stderr or stdout or "Unknown error"
            return {
                "success": False,
                "error": f"Verification failed: {error_msg}",
                "payload": None,
                "raw_output": stdout
            }

        if not stdout:
            return {
                "success": False,
                "error": "No output from verification pipeline",
                "payload": None,
                "raw_output": ""
            }

        try:
            payload_data = json.loads(stdout)
            return {
                "success": True,
                "payload": payload_data,
                "error": None,
                "raw_output": stdout
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse JSON payload: {str(e)}",
                "payload": None,
                "raw_output": stdout
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "payload": None,
            "raw_output": ""
        }

    finally:
        if type in ("binary", "string") and quote_file_path and os.path.exists(quote_file_path):
            try:
                os.unlink(quote_file_path)
            except Exception as e:
                print(f"[WARNING] Failed to delete temp quote file {quote_file_path}: {str(e)}")