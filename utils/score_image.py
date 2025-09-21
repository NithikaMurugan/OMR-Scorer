import argparse
import json
import os
from utils import bubbledetection


def main():
    parser = argparse.ArgumentParser(description="Score a single OMR image and print results")
    parser.add_argument("image", help="Path to the OMR image (jpg/png)")
    parser.add_argument("--key", dest="key", default=None, help="Path to the Excel answer key (defaults to sampledata/answer_key.xlsx.xlsx)")
    parser.add_argument("--set", dest="set_name", default="Set - A", help="Sheet name in the answer key (default: Set - A)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print full result JSON instead of summary")
    args = parser.parse_args()

    result = bubbledetection.score_omr_image(args.image, args.key, args.set_name)

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Total: {result['total']} / {result['max_possible']}  ({result['percentage']:.1f}%)")
        sections = result.get("sections", {})
        for subject_name, correct_count in sections.items():
            print(f" - {subject_name}: {correct_count}/20")


if __name__ == "__main__":
    main()
