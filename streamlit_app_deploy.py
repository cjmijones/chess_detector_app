# streamlit_app.py · updated with label abbreviations and synced visuals
import sys, io, torch, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
import numpy as np
from huggingface_hub import hf_hub_download


# ── repo paths ------------------------------------------------------
ROOT, SCRIPTS = Path(__file__).parent.resolve(), Path(__file__).parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sandbox_loader import create_val_sandbox, cleanup_root  # noqa


ckpt_path = hf_hub_download(
    repo_id="cjmijones/chess-predictor-model",
    filename="best-epoch-epoch=11-val_loss=val_loss=0.2299.ckpt",
    cache_dir="checkpoints"
)

CKPT = ckpt_path

# ── page config -----------------------------------------------------
st.set_page_config(page_title="♟️ Chess Board Predictor", layout="wide")
st.title("♟️ Chess Board Predictor — training-exact pipeline")

uploaded = st.file_uploader(
    label="Upload a chess-board photo", type=("jpg", "jpeg", "png"),
    label_visibility="collapsed"
)

# ── label mappings --------------------------------------------------
ID_TO_NAME = {
    0: "wp", 1: "wr", 2: "wn", 3: "wb", 4: "wq", 5: "wk",
    6: "bp", 7: "br", 8: "bn", 9: "bb", 10: "bq", 11: "bk",
    12: "empty"
}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}

# ── helper: render board with piece images --------------------------
def render_chessboard(label_matrix, piece_dir: Path, square_size=80):
    """
    Render an 8×8 chessboard image from a label matrix.
    """
    assert label_matrix.shape == (8, 8)

    white = (240, 217, 181)
    brown = (181, 136, 99)

    PIECE_FILE_MAP = {
        0: "white-pawn", 1: "white-rook", 2: "white-knight", 3: "white-bishop",
        4: "white-queen", 5: "white-king",
        6: "black-pawn", 7: "black-rook", 8: "black-knight", 9: "black-bishop",
        10: "black-queen", 11: "black-king"
    }

    board_img = Image.new("RGBA", (square_size * 8, square_size * 8))
    draw = ImageDraw.Draw(board_img)

    piece_imgs = {
        name: Image.open(piece_dir / f"{name}.png").resize((square_size, square_size), Image.LANCZOS)
        for name in set(PIECE_FILE_MAP.values())
    }

    for row in range(8):
        for col in range(8):
            x0, y0 = col * square_size, row * square_size
            color = white if (row + col) % 2 == 0 else brown
            draw.rectangle([x0, y0, x0 + square_size, y0 + square_size], fill=color)

            piece_id = label_matrix[row, col]
            piece_name = PIECE_FILE_MAP.get(piece_id)
            if piece_name:
                board_img.paste(piece_imgs[piece_name], (x0, y0), mask=piece_imgs[piece_name])

    return board_img

# ── main pipeline ---------------------------------------------------
if uploaded:
    img_bytes = uploaded.read()
    pil_disp = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    with st.spinner("Running model…"):
        model, val_set, idx, tmp_root = create_val_sandbox(
            img_bytes, chkpt_path=CKPT, px_resize=512
        )
        x, _ = val_set[idx]
        with torch.no_grad():
            board = (
                model(x.unsqueeze(0))
                .argmax(-1)
                .view(8, 8)
                .cpu()
                .numpy()
                .astype(int)
            )

    # ─ visual: original + board reconstruction ─
    st.markdown("### Visual Predictions")
    col_img, col_render = st.columns([1.1, 1])
    with col_img:
        st.image(pil_disp, caption="Uploaded image", width=520)

    with col_render:
        with st.spinner("Rendering board..."):
            piece_dir = Path("resources/pieces")
            board_img = render_chessboard(board, piece_dir, square_size=64)
        st.image(board_img, caption="Reconstructed board", width=520)

    # ─ visual: original prediction matrix + heatmap ─
    col_heatmap, col_tbl = st.columns([1, 1.1])
    with col_heatmap:
        fig, ax = plt.subplots(figsize=(3.12, 3.12))
        im = ax.imshow(board, cmap="viridis", vmin=0, vmax=12)
        for i in range(8):
            for j in range(8):
                label = ID_TO_NAME.get(board[i, j], "")
                if label != "empty":
                    ax.text(j, i, label, ha="center", va="center", color="white", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig, use_container_width=False)

    with col_tbl:
        st.markdown("###### Predicted label matrix")
        st.dataframe(board, height=320, width=520)

    # ─ editable prediction section ─
    st.markdown("### 🧪 Edit the Predictions")

    if "editable_board" not in st.session_state:
        st.session_state.editable_board = board.tolist()

    col_edit_heatmap, col_edit_grid = st.columns([1, 1.2])
    with col_edit_grid:
        updated_board = []
        st.markdown("##### Editable 8×8 Prediction Grid")

        DROPDOWN_OPTIONS = [ "empty", "wp", "wr", "wn", "wb", "wq", "wk", "bp", "br", "bn", "bb", "bq", "bk" ]

        with st.form("edit_matrix_form"):
            for i in range(8):
                row = []
                cols = st.columns(8)
                for j in range(8):
                    label = f"{i}-{j}"
                    current_val = st.session_state.editable_board[i][j]
                    # safely map current label
                    current_label = ID_TO_NAME.get(current_val, "empty")
                    if current_label not in DROPDOWN_OPTIONS:
                        current_label = "empty"

                    # dropdown input
                    new_label = cols[j].selectbox(
                        label,
                        options=DROPDOWN_OPTIONS,
                        index=DROPDOWN_OPTIONS.index(current_label),
                        key=f"cell-{i}-{j}",
                        label_visibility="collapsed"
                    )

                    # convert label back to ID (safe fallback)
                    row.append(NAME_TO_ID.get(new_label, 12))  # 12 = "empty"
                updated_board.append(row)
            submit = st.form_submit_button("🔁 Update Heatmap")

    if submit:
        st.session_state.editable_board = updated_board

    with col_edit_heatmap:
        fig2, ax2 = plt.subplots(figsize=(3.12, 3.12))
        edited_array = np.array(st.session_state.editable_board)
        im2 = ax2.imshow(edited_array, cmap="viridis", vmin=0, vmax=12)
        for i in range(8):
            for j in range(8):
                label = ID_TO_NAME.get(edited_array[i, j], "")
                if label != "empty":
                    ax2.text(j, i, label, ha="center", va="center", color="white", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
        st.pyplot(fig2, use_container_width=False)

    cleanup_root(tmp_root)
