from typing import List, Tuple
import numpy as np

class FENGenerator:
    @staticmethod
    def generate_fen(board_state: List[List[str]]) -> str:
        def compress_empty_squares(row: List[str]) -> str:
            compressed = ''
            empty_count = 0
            for square in row:
                if square == '1':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        compressed += str(empty_count)
                        empty_count = 0
                    compressed += square
            if empty_count > 0:
                compressed += str(empty_count)
            return compressed

        fen_rows = [compress_empty_squares(row) for row in board_state]
        return '/'.join(fen_rows)

    @staticmethod
    def create_board_state(ptsT: List[Tuple[float, float]], ptsL: List[Tuple[float, float]], 
                           detections: np.ndarray, boxes: object) -> List[List[str]]:
        x_coords = [pt[0] for pt in ptsT]
        y_coords = [pt[1] for pt in ptsL]

        board_state = []
        for i in range(8):
            row = []
            for j in range(8):
                square = np.array([
                    [x_coords[j], y_coords[i]],
                    [x_coords[j+1], y_coords[i]],
                    [x_coords[j+1], y_coords[i+1]],
                    [x_coords[j], y_coords[i+1]]
                ])
                from chess_piece_detector import ChessPieceDetector
                piece = ChessPieceDetector.connect_square_to_detection(detections, square, boxes)
                row.append(piece)
            board_state.append(row)
        return board_state