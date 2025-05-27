import socket
import pickle
import numpy as np
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class MatrixClient:
    def __init__(self, server_ports: list):
        self.server_ports = server_ports
        self.host = 'localhost'

    def send_message(self, sock, message):
        # Serializa a mensagem usando pickle para converter o objeto em uma sequência de bytes.
        data = pickle.dumps(message)
        # Empacota o tamanho da mensagem (número de bytes) como um inteiro de 4 bytes.
        # Isso é usado para que o servidor saiba quantos bytes esperar.
        length_prefix = struct.pack('!I', len(data))
        # Envia o tamanho seguido pela mensagem serializada.
        # sock.sendall garante que todos os dados sejam enviados.
        sock.sendall(length_prefix + data)

    def receive_message(self, sock):
        # Recebe os primeiros 4 bytes, que contêm o tamanho da mensagem.
        length_data = sock.recv(4)
        # Se não receber dados (ex: conexão fechada), retorna None.
        if not length_data:
            return None
        # Desempacota os 4 bytes para obter o tamanho da mensagem como um inteiro.
        message_length = struct.unpack('!I', length_data)[0]
        
        # Inicializa uma string de bytes vazia para acumular os dados recebidos.
        data = b""
        # Guarda o número de bytes restantes a serem lidos.
        remaining = message_length
        # Loop para receber os dados da mensagem em pedaços (chunks).
        while remaining > 0:
            # Tenta receber até 4096 bytes ou o número de bytes restantes, o que for menor.
            chunk = sock.recv(min(remaining, 4096))
            # Se não receber nenhum chunk (ex: conexão fechada inesperadamente), interrompe o loop.
            if not chunk:
                break
            # Adiciona o chunk recebido aos dados acumulados.
            data += chunk
            # Decrementa o número de bytes restantes.
            remaining -= len(chunk)
        
        # Verifica se todos os bytes esperados foram recebidos.
        if len(data) == message_length:
            # Desserializa os dados usando pickle para reconstruir o objeto Python original.
            return pickle.loads(data)
        # Se o número de bytes recebidos não corresponder ao esperado, retorna None (indicando erro ou mensagem incompleta).
        return None


    def compute_dot_product(self, server_port: int, vector1: np.ndarray, vector2: np.ndarray) -> float:
        # Cria um socket TCP/IP, garantindo que o socket é fechado automaticamente.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                # Conecta ao servidor no host e porta.
                sock.connect((self.host, server_port))
                # Envia os dois vetores (empacotados como uma tupla) para o servidor.
                self.send_message(sock, (vector1, vector2))
                # Recebe a resposta (o produto escalar calculado) do servidor.
                result = self.receive_message(sock)
                # Retorna o resultado do produto escalar.
                return result
            except Exception as e:
                # Se ocorrer qualquer erro durante a comunicação (ex: conexão recusada, erro ao enviar/receber),
                logging.error(f"Erro com o servidor na porta {server_port}: {str(e)}")
                return None

    def multiply_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        # Pega as dimensões das matrizes de entrada.
        rows_a, _ = matrix_a.shape
        # Pega colunas de b.
        cols_b = matrix_b.shape[1]
        # Inicializa a matriz resultado com zeros.
        result = np.zeros((rows_a, cols_b))
        
        # Cria uma lista de tarefas para calcular cada elemento da matriz resultado.
        # Cada tarefa consiste nos índices (i, j) do elemento e os vetores correspondentes (linha de A, coluna de B).
        tasks = []
        for i in range(rows_a):
            for j in range(cols_b):
                row = matrix_a[i, :]
                col = matrix_b[:, j]
                tasks.append((i, j, row, col))

        # Utiliza um ThreadPoolExecutor para fazer as tarefas em paralelo.
        # O número de workers é igual ao número de servidores disponíveis.
        with ThreadPoolExecutor(max_workers=len(self.server_ports)) as executor:
            futures = []
            # Distribui as tarefas entre os servidores.
            for task_idx, (i, j, row, col) in enumerate(tasks):
                # Seleciona a porta do servidor para esta tarefa.
                server_index = task_idx % len(self.server_ports)
                server_port = self.server_ports[server_index]
                # Manda a tarefa de calcular o produto escalar para o executor.
                future = executor.submit(self.compute_dot_product, server_port, row, col)
                # Armazena o objeto Future junto com os índices (i, j) para saber onde colocar o resultado.
                futures.append((future, i, j))

            # Coleta os resultados das tarefas à medida que são concluídas.
            for future, i, j in futures:
                try:
                    # Pega o resultado do produto escalar. `future.result()` bloqueia até que a tarefa seja concluída.
                    dot_product_result = future.result()
                    # Se o resultado não for None (ou seja, não houve erro no servidor), insere na matriz resultado.
                    if dot_product_result is not None:
                        result[i, j] = dot_product_result
                    else:
                        # Se ocorrer qualquer erro durante a comunicação (ex: conexão recusada, erro ao enviar/receber),
                        logging.error(f"Falha ao calcular o elemento ({i}, {j}) devido a erro no servidor.")
                except Exception as e:
                    # Captura exceções ao tentar pegar o resultado da future (ex: se a tarefa lançou uma exceção).
                    logging.error(f"Erro ao processar o resultado para o elemento ({i}, {j}): {str(e)}")

        # Retorna a matriz resultado.
        return result


A = np.random.randint(0, 100, size=(25, 25))
B = np.random.randint(0, 100, size=(25, 25))

SERVER_PORTS = [5000, 5001]

client = MatrixClient(SERVER_PORTS)

start_time = time.time()

result = client.multiply_matrices(A, B)

end_time = time.time()
logging.info(f"Multiplicação completada em {end_time - start_time:.4f} segundos.")

print(f"Matriz A:\n{A}")
print(f"Matriz B:\n{B}")
print(f"Resultado:\n{result}")
