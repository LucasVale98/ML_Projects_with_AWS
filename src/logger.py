import logging
import os
from datetime import datetime
 
"""Set up logging configuration"""

# Nome do arquivo de log (com data e hora)
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Caminho da PASTA logs (sem o nome do arquivo aqui)
logs_dir = os.path.join(os.getcwd(), "logs")

# Cria a pasta "logs" se não existir
os.makedirs(logs_dir, exist_ok=True)

# Caminho completo do arquivo de log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configuração do logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Opcional: teste rápido ao executar diretamente
if __name__ == "__main__":
    logging.info("Logging configurado com sucesso!")
    print(f"Logs sendo salvos em: {LOG_FILE_PATH}")
