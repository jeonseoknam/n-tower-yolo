import argparse
import sys
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import cv2
import numpy as np


# =========================
# 1) Ultralytics 경고/안내 출력 차단 설정
# =========================

# Ultralytics 로그 최소화 (환경변수는 반드시 ultralytics import 전에 설정)
os.environ.setdefault("YOLO_VERBOSE", "False")

# Ultralytics 설정 파일 생성 경로를 "항상 쓰기 가능한 곳"으로 강제
_cfg_dir = Path(tempfile.gettempdir()) / "Ultralytics"  # 예: /tmp/Ultralytics
_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_cfg_dir))


@contextmanager
def suppress_output():
    """
    stdout/stderr 를 /dev/null 로 보내서 Ultralytics의 경고/안내/로그가
    채점 stdout을 오염시키지 않게 한다.
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


# Ultralytics import 자체에서 출력이 나오는 경우가 있어서 import도 suppress
try:
    with suppress_output():
        from ultralytics import YOLO
except Exception as e:
    # 치명적 에러는 stderr로만 출력 (stdout 채점 오염 방지)
    print(f"[ERROR] Failed to import ultralytics: {e}", file=sys.stderr, flush=True)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser("CV assignment runner")
    p.add_argument("--input", required=True, type=str, help="path to input image")
    p.add_argument("--task", required=True, type=str, choices=["presence", "bbox"], help="Task to perform")
    p.add_argument("--vis_output", type=str, default=None, help="path to save visualization image (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 모델 로드 (로드 과정에서도 간혹 로그가 나올 수 있으므로 suppress)
    try:
        with suppress_output():
            model = YOLO("best.pt")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}", file=sys.stderr, flush=True)
        return 1

    # 2. 이미지 추론 (추론 과정에서도 로그/경고가 나올 수 있으므로 suppress)
    try:
        with suppress_output():
            results = model.predict(source=args.input, save=False, verbose=False, conf=0.5)
    except Exception as e:
        print(f"[ERROR] Error prediction: {e}", file=sys.stderr, flush=True)
        return 1

    # 3. 결과 분석
    detected = False
    box_info = None
    final_coords = None  # (x_min, y_min, x_max, y_max)

    try:
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]
            detected = True

            coords = best_box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = coords

            final_coords = (int(x_min), int(y_min), int(x_max), int(y_max))

            width = x_max - x_min
            height = y_max - y_min
            box_info = (int(x_min), int(y_min), int(width), int(height))
    except Exception as e:
        print(f"[ERROR] Error processing boxes: {e}", file=sys.stderr, flush=True)
        return 1

    # 4. 시각화 저장 (옵션일 때만, 그리고 stdout 출력 금지)
    if args.vis_output:
        try:
            img_vis = cv2.imread(args.input)
            if img_vis is None:
                raise RuntimeError(f"Failed to read image: {args.input}")

            if detected and final_coords:
                x1, y1, x2, y2 = final_coords
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.imwrite(args.vis_output, img_vis)
        except Exception as e:
            print(f"[ERROR] Error saving visualization: {e}", file=sys.stderr, flush=True)

    # 5. 채점용 stdout 출력 (이 줄들만 stdout으로 나가게 유지)
    if args.task == "presence":
        print("true" if detected else "false", flush=True)

    elif args.task == "bbox":
        if detected and box_info:
            print(f"{box_info[0]},{box_info[1]},{box_info[2]},{box_info[3]}", flush=True)
        else:
            print("none", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
