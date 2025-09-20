import argparse
import os
import sys
import json

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import bubbledetection
from utils import answerkey


def main():
    parser = argparse.ArgumentParser(description='Evaluate scoring on a folder of OMR images against an answer key')
    parser.add_argument('--images-root', required=True)
    parser.add_argument('--key', default=os.path.join('sampledata', 'answer_key.xlsx.xlsx'))
    parser.add_argument('--set', dest='set_name', default='Set - A')
    parser.add_argument('--summary', action='store_true', help='Print only total scores per image')
    args = parser.parse_args()

    for dirpath, _, filenames in os.walk(args.images_root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                p = os.path.join(dirpath, f)
                try:
                    result = bubbledetection.score_omr_image(p, args.key, args.set_name)
                    if args.summary:
                        print(f"{p}: {result['total']} / {result['max_possible']} ({result['percentage']:.1f}%)")
                    else:
                        print(json.dumps({'image': p, **result}, indent=2))
                except Exception as e:
                    print(f"ERROR {p}: {e}")


if __name__ == '__main__':
    main()
