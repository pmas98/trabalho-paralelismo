import socket
import pickle
import logging
import struct
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bfs_server.log'),
        logging.StreamHandler()
    ]
)

class BFSServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        logging.info(f"BFS Server initialized and listening on {self.host}:{self.port}")

    def bfs(self, g, start, end):
        if start == end:
            return [[start]]

        fila = deque([(start, [start])])
        paths = []

        while fila:
            current, path = fila.popleft()
            for vizinho in g.get(current, []):
                if vizinho in path:
                    continue
                novo_caminho = path + [vizinho]
                if vizinho == end:
                    paths.append(novo_caminho)
                else:
                    fila.append((vizinho, novo_caminho))
        return paths

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
                    # Receive subgraph and parameters from client
                    message = self.receive_message(client_socket)
                    if message is None:
                        continue

                    subgraph, start_node, end_node = message
                    logging.info(f"Received subgraph with {len(subgraph)} nodes")

                    # Compute BFS on subgraph
                    paths = self.bfs(subgraph, start_node, end_node)
                    logging.info(f"Found {len(paths)} paths")

                    # Send results back
                    self.send_message(client_socket, paths)
                    logging.info("Results sent back to client")

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
    server = BFSServer(port=port)
    server.start() 
