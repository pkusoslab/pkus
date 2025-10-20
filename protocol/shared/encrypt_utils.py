#ECDH+GCM-AES密钥函数
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

def generate_ecdh_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    pub_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_key, pub_pem.strip()

def derive_shared_key(private_key, peer_pub_pem: bytes):
    peer_key = serialization.load_pem_public_key(peer_pub_pem)
    shared_key = private_key.exchange(ec.ECDH(), peer_key)
    #使用SHA-256缩短为32字节的AES密钥
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b'ecdh-tls13',
    ).derive(shared_key)
    return derived  #返回32字节AES密钥