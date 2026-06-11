import sys


def main():
    if '--tui' in sys.argv:
        sys.argv.remove('--tui')
        from tui import main as tui_main
        tui_main()
    else:
        from cli import main as cli_main
        cli_main()


if __name__ == '__main__':
    main()
