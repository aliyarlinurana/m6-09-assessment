import sys
import os
import csv
from pathlib import Path

sys.path.insert(0, "/app")

STUDENT_JSON = "/app/STUDENT.json"
ONNX_MODEL   = "/app/models/best.onnx"
INPUT_DIR    = "/data/input"
OUTPUT_DIR   = "/data/output"
OUTPUT_CSV   = "/data/output/predictions.csv"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def cmd_info():
    with open(STUDENT_JSON, "r") as f:
        print(f.read())


def cmd_predict():
    from app.detector import CatDetector

    detector = CatDetector(ONNX_MODEL, imgsz=640, conf=0.25, class_names=("cat",))

    input_path = Path(INPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        p for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ]

    rows = []
    for img_path in sorted(image_files):
        rel_path = img_path.relative_to(input_path).as_posix()
        try:
            detections = detector.predict(str(img_path))
        except Exception as e:
            print(f"Warning: failed on {rel_path}: {e}", file=sys.stderr)
            detections = []

        if detections:
            for det in detections:
                rows.append({
                    "image_path": rel_path,
                    "xmin":       det["xmin"],
                    "ymin":       det["ymin"],
                    "xmax":       det["xmax"],
                    "ymax":       det["ymax"],
                    "confidence": det["confidence"],
                    "class":      det["class"],
                })
        else:
            rows.append({
                "image_path": rel_path,
                "xmin": "", "ymin": "", "xmax": "", "ymax": "",
                "confidence": "", "class": "",
            })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(image_files)} images processed -> {OUTPUT_CSV}")


def main():
    if len(sys.argv) < 2:
        print("Usage: cli.py [info|predict]", file=sys.stderr)
        sys.exit(1)
    command = sys.argv[1].lower()
    if command == "info":
        cmd_info()
    elif command == "predict":
        cmd_predict()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
