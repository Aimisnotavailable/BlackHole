import sys
from scripts.app import BlackHoleApp

def main():
    try:
        app = BlackHoleApp()
        app.run()
    except Exception as exc:
        print(f"Failed to start: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()