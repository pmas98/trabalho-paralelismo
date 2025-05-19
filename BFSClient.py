import socket
import pickle
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bfs_client.log'),
        logging.StreamHandler()
    ]
)

class BFSClient:
    def __init__(self, server_ports: list):
        self.server_ports = server_ports
        self.host = 'localhost'
        logging.info(f"BFS Client initialized with {len(server_ports)} servers")

    def create_subgraph(self, main_graph, start_node, neighbor):
        subgraph = {}
        visited = set()
        queue = deque([neighbor])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                subgraph[node] = [n for n in main_graph[node] if n != start_node]
                for child in main_graph[node]:
                    if child != start_node and child not in visited:
                        queue.append(child)
        return subgraph

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

    def process_subgraph(self, server_port: int, subgraph: dict, start_node: str, end_node: str) -> list:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.host, server_port))
                self.send_message(sock, (subgraph, start_node, end_node))
                paths = self.receive_message(sock)
                return paths
            except Exception as e:
                logging.error(f"Error with server on port {server_port}: {str(e)}")
                return []

    def parallel_bfs(self, graph: dict, start_node: str, end_node: str) -> tuple[list, float]:
        if start_node == end_node:
            return [[start_node]], 0.0

        inicio = time.perf_counter()
        vizinhos = graph.get(start_node, [])
        resultados = []

        with ThreadPoolExecutor(max_workers=len(self.server_ports)) as executor:
            futures = []
            for i, viz in enumerate(vizinhos):
                subgraph = self.create_subgraph(graph, start_node, viz)
                server_port = self.server_ports[i % len(self.server_ports)]
                future = executor.submit(self.process_subgraph, server_port, subgraph, viz, end_node)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    caminhos = future.result()
                    resultados.extend([[start_node] + p for p in caminhos])
                except Exception as e:
                    logging.error(f"Error processing subgraph: {str(e)}")

        fim = time.perf_counter()
        tempo_execucao = fim - inicio
        logging.info(f"Found {len(resultados)} paths in {tempo_execucao:.6f} seconds")
        return resultados, tempo_execucao

def main():
    # Example graph (same as in BFS.py)
    grafo = {
        'N0': ['N3', 'N24', 'N1', 'N7'],
        'N1': ['N6', 'N5', 'N0', 'N9', 'N13'],
        'N2': ['N14', 'N15', 'N12', 'N19', 'N21', 'N3', 'N17'],
        'N3': ['N15', 'N0', 'N13', 'N21', 'N2'],
        'N4': ['N17', 'N11', 'N8', 'N20'],
        'N5': ['N19', 'N1'],
        'N6': ['N22', 'N24', 'N1', 'N11'],
        'N7': ['N8', 'N15', 'N0', 'N19', 'N9', 'N21', 'N18'],
        'N8': ['N23', 'N13', 'N4', 'N7'],
        'N9': ['N1', 'N7'],
        'N10': ['N14', 'N11', 'N23'],
        'N11': ['N10', 'N6', 'N16', 'N4', 'N12', 'N13'],
        'N12': ['N11', 'N23', 'N2'],
        'N13': ['N11', 'N20', 'N8', 'N15', 'N3', 'N1'],
        'N14': ['N10', 'N2'],
        'N15': ['N7', 'N19', 'N13', 'N3', 'N23', 'N2'],
        'N16': ['N19', 'N11', 'N21', 'N20'],
        'N17': ['N24', 'N4', 'N2'],
        'N18': ['N22', 'N7'],
        'N19': ['N5', 'N16', 'N15', 'N7', 'N2'],
        'N20': ['N16', 'N13', 'N4'],
        'N21': ['N16', 'N7', 'N3', 'N23', 'N2'],
        'N22': ['N6', 'N18'],
        'N23': ['N10', 'N8', 'N15', 'N12', 'N21'],
        'N24': ['N6', 'N17', 'N0'],
    }

    # Server ports (you can modify these based on your setup)
    SERVER_PORTS = [5000, 5001]
    
    # Create client and run BFS
    client = BFSClient(SERVER_PORTS)
    start_node = 'N0'
    end_node = 'N24'
    
    paths, execution_time = client.parallel_bfs(grafo, start_node, end_node)
    
    print(f"\nFound {len(paths)} paths from {start_node} to {end_node}")
    print(f"Execution time: {execution_time:.6f} seconds")
    print("\nFirst few paths:")
    for path in paths[:5]:
        print(" -> ".join(path))

if __name__ == "__main__":
    main() 
