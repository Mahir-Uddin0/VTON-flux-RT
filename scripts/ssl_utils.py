import os
import socket
import subprocess


def get_lan_ip() -> str:
    """Detect the machine's primary LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 8.8.8.8 does not need to be reachable; getsockname returns the interface IP chosen by the routing table
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.103.232"


def ensure_ssl_certificates(
    cert_path: str = "certs/cert.pem",
    key_path: str = "certs/key.pem",
    lan_ip: str | None = None,
) -> tuple[str, str]:
    """Ensure SSL certificates exist with Subject Alternative Name (SAN) for LAN IP and localhost.

    If missing, automatically generates a 365-day self-signed certificate using OpenSSL.
    """
    if lan_ip is None:
        lan_ip = get_lan_ip()

    cert_dir = os.path.dirname(os.path.abspath(cert_path))
    os.makedirs(cert_dir, exist_ok=True)

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    print(f"[FluxRT SSL] Generating self-signed TLS certificate for LAN IP: {lan_ip}...", flush=True)

    san_entries = [
        f"IP:{lan_ip}",
        "IP:127.0.0.1",
        "DNS:localhost",
    ]
    if lan_ip != "192.168.103.232":
        san_entries.append("IP:192.168.103.232")

    san_str = ",".join(san_entries)

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        key_path,
        "-out",
        cert_path,
        "-days",
        "365",
        "-subj",
        f"/CN={lan_ip}/O=FluxRT Virtual Try-On",
        "-addext",
        f"subjectAltName={san_str}",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OpenSSL certificate generation failed:\n{result.stderr}")

    print(f"[FluxRT SSL] Certificate generated successfully at {cert_path} (SAN: {san_str})", flush=True)
    return cert_path, key_path


if __name__ == "__main__":
    ip = get_lan_ip()
    print("Detected LAN IP:", ip)
    c, k = ensure_ssl_certificates()
    print("Cert:", c, "Key:", k)

