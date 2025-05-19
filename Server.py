import socket
import pickle
import numpy as np
import logging
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

class DotProductServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        logging.info(f"Server initialized and listening on {self.host}:{self.port}")

    def compute_dot_product(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        return np.dot(vector1, vector2)

    def receive_message(self, client_socket):
        length_data = client_socket.recv(4)
        if not length_data:
            return None
        message_length = struct.unpack('!I', length_data)[0]
        
        data = b""
        remaining = message_length
        while remaining > 0:
            chunk = client_socket.recv(min(remaining, 4096))
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        
        if len(data) == message_length:
            return pickle.loads(data)
        return None

    def send_message(self, client_socket, message):
        data = pickle.dumps(message)
        length_prefix = struct.pack('!I', len(data))
        client_socket.sendall(length_prefix + data)

    def start(self):
        try:
            while True:
                logging.info("Waiting for client connection...")
                client_socket, address = self.server_socket.accept()
                logging.info(f"New connection established from {address}")

                try:
                    message = self.receive_message(client_socket)
                    if message is None:
                        continue

                    vector1, vector2 = message
                    logging.info(f"Received vectors of shapes {vector1.shape} and {vector2.shape}")

                    result = self.compute_dot_product(vector1, vector2)
                    logging.info(f"Computed dot product: {result}")

                    self.send_message(client_socket, result)
                    logging.info("Result sent back to client")

                except Exception as e:
                    logging.error(f"Error processing request: {str(e)}", exc_info=True)
                finally:
                    client_socket.close()
                    logging.info(f"Connection with {address} closed")

        except KeyboardInterrupt:
            logging.info("Server shutdown initiated by keyboard interrupt")
        finally:
            self.server_socket.close()
            logging.info("Server socket closed")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    server = DotProductServer(port=port)
    server.start() 
