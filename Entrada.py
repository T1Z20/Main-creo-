import numpy as np
import chess

def evaluar_seguridad_rey(board, es_blanco):
    canal_seguridad_rey = np.zeros((8, 8))
    
    # Obtener la posición del rey
    rey = board.king(es_blanco)
    if rey is None:
        return canal_seguridad_rey  # Si no hay rey, el canal queda vacío
    
    row_rey, col_rey = divmod(rey, 8)

    # Casillas alrededor del rey
    for i in range(max(0, row_rey - 1), min(8, row_rey + 2)):
        for j in range(max(0, col_rey - 1), min(8, col_rey + 2)):
            if i == row_rey and j == col_rey:
                continue
            pieza = board.piece_at(chess.square(j, i))
            if pieza:
                if pieza.color == es_blanco:
                    canal_seguridad_rey[i][j] = 1  # Defensor
                else:
                    canal_seguridad_rey[i][j] = -1  # Amenaza
    return canal_seguridad_rey

def evaluar_desequilibrio_material(board):
    canal_desequilibrio = np.zeros((8, 8))

    # Valores de las piezas según python-chess
    valores_piezas = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
        chess.ROOK: 5, chess.QUEEN: 9
    }
    
    material_blancas = 0
    material_negras = 0

    # Sumar el valor de las piezas en el tablero
    for square in chess.SQUARES:
        pieza = board.piece_at(square)
        if pieza:
            valor = valores_piezas.get(pieza.piece_type, 0)
            if pieza.color == chess.WHITE:
                material_blancas += valor
            else:
                material_negras += valor
            
    desequilibrio_material = material_blancas - material_negras
   

    # Llenar el canal con el desequilibrio de material normalizado
    canal_desequilibrio.fill(desequilibrio_material / 10)

    return canal_desequilibrio


def board_to_matrix(board):
    matrix = np.zeros((15, 8, 8))  # Cambiar a 15 canales para incluir los nuevos

    piece_map = board.piece_map()

    # Canales 1-12: Piezas
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color == chess.WHITE else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Canal 13: Jugadas válidas
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    # Canal 14: Seguridad del Rey
    canal_seguridad_rey_blancas = evaluar_seguridad_rey(board, chess.WHITE)
    canal_seguridad_rey_negras = evaluar_seguridad_rey(board, chess.BLACK)
    matrix[13] = canal_seguridad_rey_blancas + canal_seguridad_rey_negras

    # Canal 15: Desequilibrio Material
    canal_desequilibrio = evaluar_desequilibrio_material(board)
    matrix[14] = canal_desequilibrio

    return matrix

def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int