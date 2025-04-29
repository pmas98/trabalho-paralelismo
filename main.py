import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def create_subgraph(main_graph, start_node, neighbor):
    subgraph = {}                     # Inicializa o dicionário do subgrafo
    visited = set()                  # Conjunto de nós já visitados
    queue = deque([neighbor])       # Fila para BFS iniciando pelo vizinho

    while queue:                    # Enquanto houver nós na fila
        node = queue.popleft()      # Remove o próximo nó da fila
        if node not in visited:     # Se o nó ainda não foi visitado
            visited.add(node)       # Marca o nó como visitado
            # Adiciona os vizinhos do nó atual ao subgrafo, exceto o nó de origem
            subgraph[node] = [n for n in main_graph[node] if n != start_node]
            for child in main_graph[node]:   # Para cada vizinho do nó atual
                if child != start_node and child not in visited:  # Se não for o nó inicial e ainda não visitado
                    queue.append(child)      # Adiciona à fila para visitar depois
    return subgraph               # Retorna o subgrafo construído

def bfs(g, start, end):
    if start == end:                   # Edge case: início e fim são o mesmo nó
        return [[start]]              # Retorna o caminho trivial

    fila = deque([(start, [start])])  # Fila para BFS, que armazenan o nó atual e o caminho até ele
    paths = []                        # Lista de caminhos encontrados

    while fila:                       # Enquanto existirem caminhos na fila
        current, path = fila.popleft()  # Extrai o nó atual e o caminho até ele
        for vizinho in g.get(current, []):  # Para cada vizinho do nó atual
            if vizinho in path:      # Evita ciclos
                continue
            novo_caminho = path + [vizinho]  # Cria novo caminho incluindo o vizinho
            if vizinho == end:       # Se chegou ao destino, adiciona à lista de caminhos
                paths.append(novo_caminho)
            else:
                fila.append((vizinho, novo_caminho))  # Continua explorando o caminho
    return paths                     # Retorna todos os caminhos encontrados

def busca_paralela(g, start, end):
    vizinhos = g.get(start, [])       # Obtém todos os vizinhos do nó inicial
    inicio = time.perf_counter()      # Marca o tempo inicial
    resultados = []                   # Lista para armazenar os caminhos encontrados
    
    with ProcessPoolExecutor() as executor:  # Cria um pool de processos paralelos
        futures = []                  # Lista de tarefas assíncronas
        for viz in vizinhos:         # Para cada vizinho do nó inicial
            subgraph = create_subgraph(g, start, viz)  # Cria um subgrafo sem o nó inicial
            # Envia uma tarefa assíncrona para encontrar caminhos a partir desse vizinho
            futures.append(
                executor.submit(bfs, subgraph, viz, end)
            )
        
        for future in as_completed(futures):           # À medida que as tarefas finalizam
            caminhos = future.result()                 # Obtém o resultado da tarefa
            resultados.extend([[start] + p for p in caminhos])  # Adiciona o nó inicial a cada caminho
    
    fim = time.perf_counter()          # Marca o tempo final
    print(f"[Paralelo ] Encontrados {len(resultados)} caminhos em {fim - inicio:.6f} segundos.")

    return resultados, fim - inicio   # Retorna os caminhos e o tempo de execução

# Executa a busca de forma sequencial (sem paralelismo)
def busca_sequencial(g, start, end):
    inicio = time.perf_counter()         # Marca o tempo inicial
    caminhos = bfs(g, start, end)        # Executa a busca BFS normalmente
    fim = time.perf_counter()            # Marca o tempo final
    print(f"[Sequencial] Encontrados {len(caminhos)} caminhos em {fim - inicio:.6f} segundos.")
    return caminhos, fim - inicio        # Retorna os caminhos e o tempo de execução

if __name__ == '__main__':
    start_node = 'N0'
    end_node = 'N24'
    
    seq_paths, seq_time = busca_sequencial(grafo, start_node, end_node)
    par_paths, par_time = busca_paralela(grafo, start_node, end_node)

    if sorted(seq_paths) == sorted(par_paths):
        print("\nCaminhos são iguais nas buscas sequencial e paralela.")
    else:
        print("\nCaminhos diferentes entre busca sequencial e paralela.")

    ganho = seq_time / par_time if par_time > 0 else float('inf')

    print(f"\nGanho de desempenho: {ganho:.2f}x mais rápido")
