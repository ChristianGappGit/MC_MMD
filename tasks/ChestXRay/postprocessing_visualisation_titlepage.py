import os
import sys


def main(perform_dir: str):
    #create title html file that has urls to "train", "val", and "test" directories
    #these directories have interpretability.html files that have urls to the individual html files

    print("creating title page for interpretability...")

    with open(os.path.join(perform_dir, 'interpretability.html'), 'w') as f:
        f.write('<h1>Interpretability</h1>\n')
        f.write('<a href="train/interpretability.html">train</a><br>\n')
        f.write('<a href="val/interpretability.html">val</a><br>\n')
        f.write('<a href="test/interpretability.html">test</a><br>\n')


if __name__ == "__main__":
    main(sys.argv[1])