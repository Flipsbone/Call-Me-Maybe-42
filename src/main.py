import sys
from parser import DataParser


def main() -> None:
    try:
        data = DataParser.load_functions("data.json")
        print("ok")
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
