import torch
import pytorch_lightning as pl
from chess_resnext import ChessResNeXt
from chess_data_module import ChessDataModule
from typing import List, Any
import numpy as np

def load_model(checkpoint_path: str) -> ChessResNeXt:
    model = ChessResNeXt.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def make_predictions(model: ChessResNeXt, data_module: ChessDataModule) -> List[np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predict_loader = data_module.predict_dataloader()

    all_predictions: List[np.ndarray] = []
    
    with torch.no_grad():
        for batch in predict_loader:
            x = batch.to(device)
            logits = model(x)
            
            probs = torch.sigmoid(logits)
            probs = probs.reshape(-1, 64, 13)
            predictions = torch.argmax(probs, dim=2)
            
            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

def board_state_to_fen(board_state: np.ndarray) -> str:
    piece_map = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k',
        12: '.'
    }
    
    fen: List[str] = []
    for row in range(8):
        empty = 0
        row_fen: List[str] = []
        for col in range(8):
            piece = piece_map[board_state[row * 8 + col]]
            if piece == '.':
                empty += 1
            else:
                if empty > 0:
                    row_fen.append(str(empty))
                    empty = 0
                row_fen.append(piece)
        if empty > 0:
            row_fen.append(str(empty))
        fen.append(''.join(row_fen))
    
    return '/'.join(fen)

def main() -> None:
    model = load_model("checkpoint.ckpt")

    data_module = ChessDataModule(
        dataroot="data/dataset",
        batch_size=32,
        workers=4,
        unlabeled_image_dir="data/unlabeled"
    )
    data_module.setup("predict")

    predictions = make_predictions(model, data_module)

    for i, pred in enumerate(predictions):
        print(f"Prediction for image {i}:")
        print(board_state_to_fen(pred))
        print()

if __name__ == "__main__":
    main()