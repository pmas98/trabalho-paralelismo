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

def bfs(g, start, end):
    fila = deque([[start]])
    caminhos = []
    while fila:
        caminho = fila.popleft()
        no = caminho[-1]
        if no == end:
            caminhos.append(caminho)
        else:
            for vizinho in g.get(no, []):
                if vizinho not in caminho:
                    fila.append(caminho + [vizinho])
    return caminhos

def busca_paralela(g, start, end):
    vizinhos = g.get(start, [])
    inicio = time.perf_counter()
    resultados = []
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for viz in vizinhos:
            subgraph = create_subgraph(g, start, viz)
            futures.append(
                executor.submit(bfs, subgraph, viz, end)
        )
        
        for future in as_completed(futures):
            caminhos = future.result()
            resultados.extend([ [start] + p for p in caminhos ])
    
    fim = time.perf_counter()
    print(f"[Paralelo ] Encontrados {len(resultados)} caminhos em {fim - inicio:.6f} segundos.")

    return resultados, fim - inicio

def busca_sequencial(g, start, end):
    inicio = time.perf_counter()
    caminhos = bfs(g, start, end)
    fim = time.perf_counter()
    print(f"[Sequencial] Encontrados {len(caminhos)} caminhos em {fim - inicio:.6f} segundos.")
    return caminhos, fim - inicio

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
