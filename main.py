import argparse

from predictors.classification import classification


def main():
    parser = argparse.ArgumentParser(description='dcop')
    parser.set_defaults(func=classification)
    parser.add_argument("generate_raw_data", type=str, help="Generate Datasets")

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
