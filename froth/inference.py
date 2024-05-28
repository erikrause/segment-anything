import os
from argparse import ArgumentParser
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from extract_markers import extract_markers
import glob


def inference(input_dir: str,
              output_dir: str,
              sam_predictor: SamPredictor,
              markers_dir: str = None,
              highlights_dir: str = None,
              preprocessed_dir: str = None):
    frames_pathes = glob.glob(f"{input_dir}/*.png")

    j = 0
    for frame_path in frames_pathes:
        print(j)
        frame = cv2.imread(frame_path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markers_out_path = f"{markers_dir}/{j}-markers.png"
        highlights_out_path = f"{highlights_dir}/{j}-highlights.png"
        preprocessed_out_path = f"{preprocessed_dir}/{j}-preprocessed.png"
        markers = extract_markers(gray_frame,
                                  markers_out_path,
                                  highlights_out_path,
                                  preprocessed_out_path)
        #labels = np.ones((markers.shape[0],))
        labels = np.arange(markers.shape[0] + 1)[1:]

        sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, scores, logits = sam_predictor.predict(
            point_coords=markers,
            point_labels=labels,
            multimask_output=True,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            out_mask = np.zeros(mask.shape, dtype=np.uint8)
            out_mask[mask] = 255
            out_mask = np.expand_dims(out_mask, axis=-1)
            cv2.imwrite(f"{output_dir}/{j}-SAM_mask-score-{score}.png", out_mask)

        j += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="frames", help="input frames folder")
    parser.add_argument("--output_dir", type=str, default="masks", help="output SAM masks predictions folder")
    parser.add_argument("--markers_dir", type=str, default=None, help="output markers folder (optional, if not set markers will not be saved)")
    parser.add_argument("--highlights_dir", type=str, default=None, help="output highlights folder (optional, if not set highlights will not be saved)")
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="output preprocessed folder (optional, if not set preprocessed images will not be saved)")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/sam_vit_h_4b8939.pth", help="SAM checkpoint path")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM type")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = "cuda"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.markers_dir is not None:
        os.makedirs(args.markers_dir, exist_ok=True)
    if args.highlights_dir is not None:
        os.makedirs(args.highlights_dir, exist_ok=True)
    if args.preprocessed_dir is not None:
        os.makedirs(args.preprocessed_dir, exist_ok=True)

    inference(args.input_dir,
              args.output_dir,
              sam_predictor,
              args.markers_dir,
              args.highlights_dir,
              args.preprocessed_dir)
