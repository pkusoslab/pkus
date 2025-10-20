# Secure Multi-Client Protocol (SMCP) - TDX Edition  
*An integral component of the Professional Knowledge Utilization System (PKUS)*

This project implements the secure data transmission layer of the **Secure Multi-Client Protocol (SMCP)**, a core communication protocol within the **Professional Knowledge Utilization System (PKUS)**. Built on **Intel TDX hardware security**, SMCP enables trusted, confidential, and mutually authenticated data exchange between multiple clients and TEE-protected knowledge services.

The current implementation focuses on the **data transfer phase**, providing end-to-end encrypted streaming via AES-GCM, secure session establishment using ECDH, and mutual remote attestation through Intel TDX Quote verification. It is designed to support concurrent, secure access to sensitive professional knowledge assets.

This is the **Intel TDX-based version** of SMCP. Other essential protocol phases — including **session initiation, capability negotiation, agreement finalization, dynamic modification, graceful exit, and access revocation** — are under active development. Additional versions targeting alternative TEE platforms (e.g., AMD SEV, TrustZone, Huawei Kunpeng) will be released to support heterogeneous PKUS deployments.
