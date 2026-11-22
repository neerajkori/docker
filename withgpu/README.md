# WSL2 + Intel Arc GPU: LLM on the GPU and use it from Docker

Step-by-step guide for setting up an Intel Arc GPU. So you can run LLMs and other processes on it from WSL2 (Ubuntu) and from Docker containers.

Reference machine this was written on:

- Host: `IN-CS7MR24`
- Manufacturer / Model: `Dell Inc. / Dell Pro 14 PC14250`
- CPU: `Intel64 Family 6 Model 181 Stepping 0 GenuineIntel` (~2000 MHz, 64-bit)
- GPU: Intel Arc integrated graphics (this is the "GPU" on this PC — it shares system RAM)
- Windows: Microsoft Windows 11 Enterprise, 10.0.26200 (build 26200)
- RAM: ~15,829 MB total (~5,048 MB available when captured)
- WSL2 (this instance): Ubuntu 24.04.5 LTS, kernel 6.18.33.2-microsoft-standard-WSL2, gcc 13.3.0
- Docker Engine inside this WSL: 29.10.0

The same steps apply on any Ubuntu 24.04 WSL2 on Intel Core Ultra-class machines.

---

## 1. Drivers to install

### 1.1 Windows side — the Intel Arc & Iris Xe Graphics driver

- Download and install the latest **Intel Arc & Iris Xe Graphics Windows driver** from the Intel
  Download Center ("Intel Arc & Iris Xe Graphics" Windows driver page).
- There is **no separate "WSL driver"** to install: WSL2 reaches the GPU through the Windows Intel
  driver (exposed as `/dev/dxg`). Updating to the latest driver on Windows is the step that
  provides the WSL2 GPU support.
- If a Windows restart is requested, do it.

### 1.2 Verify the GPU is visible in WSL2

In WSL (Ubuntu 24.04):

```sh
ls -la /dev/dxg
```

You should see a device node like:

```
crw-rw-rw- 1 root root 10, 258 ... /dev/dxg
```

`/dev/dxg` is how the Arc GPU is reached from WSL2.

### 1.3 WSL side — build tools

```sh
sudo apt update
sudo apt install -y build-essential cmake ninja-build clinfo
```

### 1.4 WSL side — Intel oneAPI Base Toolkit

This provides the Level Zero runtime, the `icx`/`icpx` SYCL compilers, and oneDNN/oneMKL, which
llama.cpp needs to run on Intel GPUs.

Install from the Intel apt repository:

```sh
Reference: - Reference: [Intel oneAPI Toolkit download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)

# download the key to system keyring
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# add signed entry to apt sources and configure the APT client to use Intel repository:
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
```

Optional for native Linux driver debugging:

```sh
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# log out / re-login after these (not required for WSL2, which uses /dev/dxg)
```

### 1.5 Verify oneAPI sees the Arc

```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

You should see the Arc listed as a Level Zero GPU, for example:

```
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) ...
```

If nothing shows up, re-check that the Windows driver from step 1.1 is the latest and that you have
re-logged in (or restarted).

---

## 2. Docker installation inside the WSL (Ubuntu 24.04)

### 2.1 Option B — Docker Engine from Docker's repository (recommended)

```sh
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo service docker start
```

### 2.2 Allow your user to run Docker (no sudo)

```sh
sudo usermod -aG docker $USER
# Close and reopen this terminal, or run: newgrp docker
```

### 2.3 Verify

```sh
docker --version
docker run --rm hello-world
```

---

## 3. Run an LLM / process on the Arc GPU in WSL (llama.cpp, SYCL backend)

This is the standard, supported way to run LLMs on the Intel Arc from WSL2.

### 3.1 Download a quantized GGUF model

```sh
mkdir -p ~/llm && cd ~/llm
wget -O qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The Arc shares system RAM, so keep models around 8–10 GB or smaller for comfortable use
(~7B models at Q4_K_M or Q4_0, 3B/1B models, etc.). Other small options on the same page:
`qwen2.5-1.5b-instruct-q4_0.gguf`, `Llama-3.2-1B-Instruct-Q4_0.gguf`, etc.

### 3.2 Build llama.cpp with the SYCL backend

```sh
cd ~/llama.cpp            # after: git clone https://github.com/ggml-org/llama.cpp
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build -j
./build/bin/llama-ls-sycl-device
```

The build takes a few minutes. `llama-ls-sycl-device` lists the SYCL devices (the Arc should be
listed, e.g. `[ext_oneapi_level_zero:gpu:0] ...`).

### 3.3 Run inference on the GPU (`-ngl 99` = offload all layers to the GPU)

```sh
./build/bin/llama-cli -m ~/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -p "Hello, how are you?" -n 128
```

### 3.4 Serve it as an OpenAI-compatible HTTP API (for use by Docker containers)

```sh
./build/bin/llama-server -m ~/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 --host 0.0.0.0 --port 8080
```

---

## 4. "Use the GPU inside the Docker container" — the practical way

Honest version of this: unlike NVIDIA (which has `nvidia-container-toolkit` with `--gpus all`),
Intel does not provide a Docker GPU-passthrough toolkit, and Intel's own Docker+SYCL images
(`ghcr.io/ggml-org/llama.cpp:*-intel`) state "WSL support has not been verified yet" and pass
`/dev/dri`, which does not exist in this WSL2.

So the two reliable patterns are:

### 4.1 Option A (recommended — works on WSL2 today): run the model server on the WSL host; containers connect to it

Start the server on the WSL host (this uses the GPU):

```sh
cd ~/llama.cpp
./build/bin/llama-server -m ~/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 --host 0.0.0.0 --port 8080
```

Then from a Docker container, use the server:

- **Docker Desktop for Windows**: the container reaches the WSL host through `host.docker.internal`:
  ```sh
  docker run -p 8080:8080 <image> curl -s http://host.docker.internal:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen2.5-1.5b-instruct-q4_k_m","messages":[{"role":"user","content":"Hello!"}]}'
  ```
  (The container can also use `-v $PWD/scripts:/scripts` for your own process code.)
- **Docker Engine running inside this WSL**: use `--network host` so the container shares the WSL
  network namespace and reaches the server via `127.0.0.1`:
  ```sh
  docker run --network host <image> curl -s http://127.0.0.1:8080/v1/chat/completions ...
  ```

### 4.2 Option B (experimental — native Linux only, not verified for WSL2): Intel SYCL Docker images

From the llama.cpp repository:

```sh
docker build -t local/llama.cpp:server-intel --target server -f .devops/intel.Dockerfile .
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 -v ~/llm:/llm -p 8080:8080 \
  local/llama.cpp:server-intel -m /llm/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99
```

Check `ls /dev/dri` first — on this WSL2 it is empty, so this option will not work here.

### 4.3 Option C (experimental): Intel ipex-llm WSL Docker variant

```sh
docker run --privileged --device /dev/dri -v /usr/lib/wsl:/usr/lib/wsl \
  intelanalytics/ipex-llm-serving-xpu --entrypoint /bin/bash
```

Inside the container, `sycl-ls` should show the Arc if passthrough works. Note the ipex-llm
project is archived (see section 5).

---

## 5. Alternative LLM stack: ipex-llm (pip) — note: archived project

If you prefer the pip-based Intel stack instead of llama.cpp:

```sh
pip install ipex-llm[all]
```

Use its PyTorch API (`ipex_llm.transformers.AutoModelForCausalLM`) or its Ollama portable zip.
Be aware that the ipex-llm project is archived as of 2026 (Intel no longer maintains it); the
llama.cpp path in section 3 is the better long-term choice.

---

## 6. Verification checklist

Run these in order; everything should pass:

```sh
# 1. GPU visible in WSL
ls -la /dev/dxg

# 2. oneAPI sees the Arc
source /opt/intel/oneapi/setvars.sh
sycl-ls            # expect: [ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) ...

# 3. llama.cpp lists the GPU
./build/bin/llama-ls-sycl-device

# 4. Real generation on the GPU (not just compile)
./build/bin/llama-cli -m ~/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf -ngl 99 -p "test" -n 32

# 5. Docker works
docker --version
docker run --rm hello-world

# 6. From a container, the model server is reachable
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-instruct-q4_k_m","messages":[{"role":"user","content":"Hello!"}]}'
```

(The server from 3.4 must be running when you run the last check; with Docker Engine in WSL use
`--network host` containers; with Docker Desktop for Windows use `http://host.docker.internal:8080`.)

---

## 7. Porting to other instances

- On the other Windows machine: do step 1.1 (latest Intel Arc & Iris Xe Graphics driver) first.
- All other steps (1.2–6) are identical for any Ubuntu 24.04 WSL2 instance on Intel Core Ultra-class
  PCs. Verify with `hostname` and `ls -la /dev/dxg`.
