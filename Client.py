import socket
import pickle
import numpy as np
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)

class MatrixClient:
    def __init__(self, server_ports: list):
        self.server_ports = server_ports
        self.host = 'localhost'
        logging.info(f"MatrixClient initialized with {len(server_ports)} servers")

    def send_message(self, sock, message):
        data = pickle.dumps(message)
        length_prefix = struct.pack('!I', len(data))
        sock.sendall(length_prefix + data)

    def receive_message(self, sock):
        length_data = sock.recv(4)
        if not length_data:
            return None
        message_length = struct.unpack('!I', length_data)[0]
        
        data = b""
        remaining = message_length
        while remaining > 0:
            chunk = sock.recv(min(remaining, 4096))
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        
        if len(data) == message_length:
            return pickle.loads(data)
        return None

    def compute_dot_product(self, server_port: int, vector1: np.ndarray, vector2: np.ndarray) -> float:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.host, server_port))
                self.send_message(sock, (vector1, vector2))
                result = self.receive_message(sock)
                return result
            except Exception as e:
                logging.error(f"Error with server on port {server_port}: {str(e)}")
                return None

    def multiply_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        rows_a, cols_a = matrix_a.shape
        cols_b = matrix_b.shape[1]
        result = np.zeros((rows_a, cols_b))
        
        tasks = []
        for i in range(rows_a):
            for j in range(cols_b):
                row = matrix_a[i, :]
                col = matrix_b[:, j]
                tasks.append((i, j, row, col))

        with ThreadPoolExecutor(max_workers=len(self.server_ports)) as executor:
            futures = []
            for task_idx, (i, j, row, col) in enumerate(tasks):
                server_port = self.server_ports[task_idx % len(self.server_ports)]
                future = executor.submit(self.compute_dot_product, server_port, row, col)
                futures.append((future, i, j))

            for future, i, j in futures:
                try:
                    result[i, j] = future.result()
                except Exception as e:
                    logging.error(f"Error computing element ({i}, {j}): {str(e)}")

        return result

def main():
    A = np.array([[2, 2], [3, 1]])
    B = np.array([[1, 2], [3, 4]])
    
    SERVER_PORTS = [5000, 5001]
    
    client = MatrixClient(SERVER_PORTS)
    result = client.multiply_matrices(A, B)
    
    expected = np.matmul(A, B)
    is_correct = np.allclose(result, expected)
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Result is {'correct' if is_correct else 'incorrect'}")

if __name__ == "__main__":
    main() 
