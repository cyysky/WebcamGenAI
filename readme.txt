pip install pytorch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

uvicorn demo_phi_3:app --host 0.0.0.0 --port 16986 --ssl-keyfile=key.pem --ssl-certfile=cert.pem

hostname -I
netsh interface portproxy add v4tov4 listenport=16986 listenaddress=0.0.0.0 connectport=16986 connectaddress=172.22.214.45
sudo ufw allow 16986

window firewall enable the port 16986