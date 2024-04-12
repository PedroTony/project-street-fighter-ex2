class Player:
    #jogador 1 é o player da esquerda, player 2 é o jogador a direita.
    left : bool
    right : bool

    #construtor
    def __init__(self, left, right):
        self.left = left
        self.right = right