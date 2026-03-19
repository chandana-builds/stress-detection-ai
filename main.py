import os

port = int(os.environ.get("PORT", 8501))

os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")
