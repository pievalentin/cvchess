
from corner_detector import CornerDetector
from image_transformer import ImageTransformer
from chessboard_analyzer import ChessboardAnalyzer
from chess_piece_detector import ChessPieceDetector
from fen_generator import FENGenerator

def analyze_chess_image(image_path: str, corner_model_path: str, piece_model_path: str) -> str:
    # Detect corners
    corner_detector = CornerDetector(corner_model_path)
    corners = corner_detector.detect_corners(image_path)

    # Transform image
    transformer = ImageTransformer()
    transformed_image = transformer.four_point_transform(image_path, corners)

    # Analyze chessboard
    analyzer = ChessboardAnalyzer()
    ptsT, ptsL = analyzer.plot_grid_on_transformed_image(transformed_image)

    # Detect chess pieces
    piece_detector = ChessPieceDetector(piece_model_path)
    detections, boxes = piece_detector.detect_pieces(transformed_image)

    # Generate FEN
    fen_generator = FENGenerator()
    board_state = fen_generator.create_board_state(ptsT, ptsL, detections, boxes)
    fen = fen_generator.generate_fen(board_state)

    return f"https://lichess.org/analysis/{fen}"

if __name__ == "__main__":
    images = ['../ex_3.jpeg'] # , 'ex_3.jpeg', 'ex_4.jpeg', 'ex_5.jpeg', 'ex_6.jpeg']
    corner_model_path = "../chess_biased.pt"
    piece_model_path = "../pieces_biased.pt"

    for image_path in images:
        lichess_url = analyze_chess_image(image_path, corner_model_path, piece_model_path)
        print(f"Analysis URL for {image_path}: {lichess_url}")