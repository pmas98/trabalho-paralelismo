import socket
import pickle
import numpy as np
import logging
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
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

    def receive_message(self, client_socket):
        # Recebe os primeiros 4 bytes que contêm o tamanho da mensagem
        length_data = client_socket.recv(4)
        if not length_data:
            # Se nenhum dado for recebido, significa que o cliente fechou a conexão
            return None
        # Desempacota os 4 bytes para obter o tamanho da mensagem (inteiro sem sinal, ordem de bytes de rede)
        message_length = struct.unpack('!I', length_data)[0]
        
        data = b""
        remaining = message_length
        # Loop para receber toda a mensagem
        while remaining > 0:
            # Recebe dados em blocos (até 4096 bytes ou o restante)
            chunk = client_socket.recv(min(remaining, 4096))
            if not chunk:
                # Se nenhum bloco for recebido, a conexão foi interrompida
                break
            data += chunk
            remaining -= len(chunk)
        
        # Verifica se a mensagem completa foi recebida
        if len(data) == message_length:
            # Desserializa os bytes recebidos usando pickle
            return pickle.loads(data)
        # Se a mensagem completa não foi recebida, retorna nada
        return None

    def send_message(self, client_socket, message):
        # Serializa o objeto da mensagem usando pickle
        data = pickle.dumps(message)
        # Empacota o tamanho dos dados serializados como um inteiro de 4 bytes sem sinal (ordem de bytes de rede)
        length_prefix = struct.pack('!I', len(data))
        # Envia o prefixo de tamanho seguido pelos dados reais
        client_socket.sendall(length_prefix + data)


    def compute_dot_product(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        # Faz a multiplicação escalar entre os dois vetores
        return np.dot(vector1, vector2)
    
    def start(self):
        try:
            # Loop infinito para aceitar conexões de clientes continuamente
            while True:
                logging.info("Aguardando conexão do cliente...")
                # Aceita uma nova conexão do cliente
                client_socket, address = self.server_socket.accept()
                logging.info(f"Nova conexão estabelecida de {address}")

                try:
                    # Recebe vetores do cliente
                    message = self.receive_message(client_socket)
                    # Se nenhuma mensagem for recebida (cliente desconectou), continua para a próxima iteração
                    if message is None:
                        continue

                    vector1, vector2 = message
                    logging.info(f"Vetores recebidos com shapes {vector1.shape} e {vector2.shape}")

                    # Calcula o produto escalar
                    result = self.compute_dot_product(vector1, vector2)
                    logging.info(f"Produto escalar calculado: {result}")

                    # Envia o resultado de volta para o cliente
                    self.send_message(client_socket, result)
                    logging.info("Resultado enviado de volta para o cliente")

                except Exception as e:
                    # Registra qualquer erro que ocorra durante o processamento da solicitação
                    logging.error(f"Erro ao processar solicitação: {str(e)}", exc_info=True)
                finally:
                    # Garante que o socket do cliente seja fechado
                    client_socket.close()
                    logging.info(f"Conexão com {address} fechada")

        except KeyboardInterrupt:
            # Lida com a interrupção do servidor via teclado (Ctrl+C)
            logging.info("Desligamento do servidor iniciado por interrupção de teclado")
        finally:
            # Garante que o socket do servidor seja fechado ao finalizar
            self.server_socket.close()
            logging.info("Socket do servidor fechado")


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    server = DotProductServer(port=port)
    server.start() 
