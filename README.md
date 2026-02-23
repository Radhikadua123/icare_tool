# icare_tool

Run the ICARE Streamlit demo with a local Ollama model (e.g. Llama 3.1).

## Setup

### 1. Conda environment

```bash
conda create -n icare python=3.10 -y
conda activate icare
cd /path/to/icare_tool
pip install -r requirements_demo.txt
```

### 2. Start Ollama server (with GPU)

From the project directory:

```bash
cd /path/to/icare_tool && OLLAMA_GPU_LAYERS=-1 ollama serve &
```

### 3. Pull the model

```bash
cd /path/to/icare_tool && ollama pull llama3.1
```

### 4. (Optional) Test the model

```bash
cd /path/to/icare_tool && time ollama run llama3.1 "What is 2+2? Answer in one word."
```

## Run the Streamlit demo

```bash
cd /path/to/icare_tool
streamlit run demo_app.py --server.port 1509
```

Open the URL shown in the terminal (e.g. `http://localhost:1509`).

## Deploy with public access (ngrok)

To expose the app via a public URL:

1. [Install ngrok](https://ngrok.com/download) and add your auth token:

   ```bash
   ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
   ```

2. With the Streamlit app running on port 1509:

   ```bash
   ngrok http 1509
   ```

Use the HTTPS URL printed by ngrok to access the demo from outside your machine.
