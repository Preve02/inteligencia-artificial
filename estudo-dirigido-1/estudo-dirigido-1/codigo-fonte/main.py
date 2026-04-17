# PROBLEMA LIGHTS OUT
# Samuel Prevedello dos Santos de Jesus

import time              # Usado para medir o tempo de execução
import random            # Usado para gerar estados aleatórios
import heapq             # Usado para fila de prioridade (A* e Gulosa)
from collections import deque  # Estrutura de fila eficiente (BFS)
import tracemalloc       # Usado para medir uso de memória



# Classe que representa o problema

class LightsOut:
    def __init__(self, n):
        # n = tamanho do tabuleiro (NxN)
        self.n = n

        # Gera automaticamente um estado inicial aleatório
        self.estado_inicial = self.gerar_estado()

    def gerar_estado(self):
        """
        Cria um estado inicial válido.
        Começa com todas as luzes ligadas (1)
        e aplica vários cliques aleatórios para embaralhar.
        """

        # Cria uma matriz NxN com todos valores = 1
        estado = [[1 for _ in range(self.n)] for _ in range(self.n)]

        # Aplica cliques aleatórios para bagunçar o tabuleiro
        for _ in range(self.n * self.n):
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)

            # Aplica a ação de clicar na posição (i, j)
            estado = self.toggle(estado, i, j)

        # Converte para tupla (imutável) para poder usar em sets/dicionários
        return tuple(tuple(linha) for linha in estado)

    def objetivo(self, estado):
        """
        Verifica se todas as luzes estão acesas (valor 1).
        Se estiverem, atingimos o objetivo.
        """
        return all(cell == 1 for linha in estado for cell in linha)

    def toggle(self, estado, i, j):
        """
        Simula o clique em uma posição (i, j).
        Ao clicar:
        - A própria célula muda
        - Seus vizinhos (cima, baixo, esquerda, direita) também mudam
        """

        # Cria uma cópia do estado atual (para não alterar o original)
        novo = [list(linha) for linha in estado]

        # Lista de posições que serão afetadas pelo clique
        for di, dj in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i + di, j + dj

            # Verifica se a posição está dentro do tabuleiro
            if 0 <= ni < self.n and 0 <= nj < self.n:

                # Inverte o valor (0 vira 1, 1 vira 0)
                novo[ni][nj] = 1 - novo[ni][nj]

        # Retorna o novo estado convertido para tupla
        return tuple(tuple(linha) for linha in novo)

    def vizinhos(self, estado):
        """
        Gera todos os estados possíveis a partir do estado atual.
        Cada posição do tabuleiro representa uma ação possível.
        """

        lista = []

        # Percorre todas as posições do tabuleiro
        for i in range(self.n):
            for j in range(self.n):

                # Gera um novo estado ao clicar nessa posição
                novo = self.toggle(estado, i, j)

                # Armazena (estado, ação realizada)
                lista.append((novo, (i, j)))

        return lista

    def heuristica(self, estado):
        """
        Função heurística:
        Conta quantas luzes estão apagadas (valor 0).

        Quanto menor esse valor, mais perto estamos do objetivo.
        """
        return sum(cell == 0 for linha in estado for cell in linha)

# ALGORITMOS DE BUSCA
def bfs(problema):
    """
    Busca em Largura (Breadth-First Search)
    Garante a solução ótima (menor número de passos),
    mas consome muita memória.
    """

    inicio = problema.estado_inicial

    # Fila: guarda (estado atual, caminho até ele)
    fila = deque([(inicio, [])])

    # Conjunto de estados já visitados
    visitados = {inicio}

    nos = 0  # Contador de nós expandidos

    while fila:
        estado, caminho = fila.popleft()
        nos += 1

        # Verifica se chegou no objetivo
        if problema.objetivo(estado):
            return caminho, nos

        # Explora vizinhos
        for prox, acao in problema.vizinhos(estado):
            if prox not in visitados:
                visitados.add(prox)
                fila.append((prox, caminho + [acao]))

    return None, nos


def dfs(problema, limite=20):
    """
    Busca em Profundidade (Depth-First Search)
    Usa um limite para evitar loops infinitos.
    Não garante solução ótima.
    """

    pilha = [(problema.estado_inicial, [], 0)]
    visitados = set()
    nos = 0

    while pilha:
        estado, caminho, prof = pilha.pop()
        nos += 1

        if problema.objetivo(estado):
            return caminho, nos

        # Só continua se não atingiu o limite de profundidade
        if prof < limite:
            for prox, acao in problema.vizinhos(estado):
                if prox not in visitados:
                    visitados.add(prox)
                    pilha.append((prox, caminho + [acao], prof + 1))

    return None, nos


def gulosa(problema):
    """
    Busca Gulosa (Greedy)
    Escolhe sempre o estado que parece melhor pela heurística.
    Rápida, mas não garante solução ótima.
    """

    inicio = problema.estado_inicial

    # Fila de prioridade baseada na heurística
    heap = [(problema.heuristica(inicio), inicio, [])]

    visitados = set()
    nos = 0

    while heap:
        h, estado, caminho = heapq.heappop(heap)
        nos += 1

        if problema.objetivo(estado):
            return caminho, nos

        visitados.add(estado)

        for prox, acao in problema.vizinhos(estado):
            if prox not in visitados:
                heapq.heappush(heap, (problema.heuristica(prox), prox, caminho + [acao]))

    return None, nos


def a_estrela(problema):
    """
    Algoritmo A*
    Combina custo real (g) + heurística (h)
    Geralmente o melhor equilíbrio entre custo e qualidade.
    """

    inicio = problema.estado_inicial

    # (f, g, estado, caminho)
    heap = [(problema.heuristica(inicio), 0, inicio, [])]

    visitados = {}
    nos = 0

    while heap:
        f, g, estado, caminho = heapq.heappop(heap)
        nos += 1

        if problema.objetivo(estado):
            return caminho, nos

        # Evita revisitar estados com custo maior
        if estado in visitados and visitados[estado] <= g:
            continue

        visitados[estado] = g

        for prox, acao in problema.vizinhos(estado):
            novo_g = g + 1
            f = novo_g + problema.heuristica(prox)

            heapq.heappush(heap, (f, novo_g, prox, caminho + [acao]))

    return None, nos


def hill_climbing(problema):
    """
    Hill Climbing (Subida de Encosta)
    Sempre escolhe o melhor vizinho imediato.
    Pode travar em ótimos locais.
    """

    estado = problema.estado_inicial
    caminho = []
    nos = 0

    while True:
        nos += 1

        if problema.objetivo(estado):
            return caminho, nos

        vizinhos = problema.vizinhos(estado)

        # Escolhe o melhor vizinho baseado na heurística
        melhor, acao = min(vizinhos, key=lambda x: problema.heuristica(x[0]))

        # Se não houver melhora, para
        if problema.heuristica(melhor) >= problema.heuristica(estado):
            return None, nos

        estado = melhor
        caminho.append(acao)


# FUNÇÃO PARA MEDIR DESEMPENHO


def executar(nome, func, problema):
    """
    Executa um algoritmo e mede:
    - tempo
    - memória
    - número de nós
    """

    tracemalloc.start()
    inicio = time.time()

    caminho, nos = func(problema)

    tempo = time.time() - inicio
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "alg": nome,
        "sucesso": caminho is not None,
        "passos": len(caminho) if caminho else 0,
        "nos": nos,
        "tempo": round(tempo, 4),
        "memoria": round(pico / 1024, 2)
    }


# PROGRAMA PRINCIPAL

def main():
    tamanhos = [2, 3, 4]

    for n in tamanhos:
        print(f"\n=== Tabuleiro {n}x{n} ===")

        problema = LightsOut(n)

        algoritmos = [
            ("BFS", bfs),
            ("DFS", dfs),
            ("Gulosa", gulosa),
            ("A*", a_estrela),
            ("Hill Climbing", hill_climbing)
        ]

        print(f"{'Algoritmo':<15}{'OK':<5}{'Passos':<8}{'Nós':<8}{'Tempo':<10}{'Mem(KB)'}")

        for nome, func in algoritmos:
            r = executar(nome, func, problema)

            print(f"{nome:<15}{str(r['sucesso']):<5}{r['passos']:<8}{r['nos']:<8}{r['tempo']:<10}{r['memoria']}")


# Executa o programa
if __name__ == "__main__":
    random.seed(42)  # Garante reprodutibilidade
    main()
