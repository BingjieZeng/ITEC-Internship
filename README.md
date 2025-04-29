This repository contains the data and source code for Bingjie Zeng's internship at the ITEC Research Institute.

- `corpora`: this directory contains the corpora used and created during the internship
    - `corpora/{corpus_name}`: each subdirectory contains a specific corpus
        - `corpora/{corpus_name}/{text}`: inside each corpus directory, there is a colleciton of texts
- `src`: this directory contains the source code created and used during the internship
    - `src/generation`: this subdirectory contains the code used for generating texts
    - `src/features`: this subdirectory contains the code used for extracting the features

Notes: Please refer to @tack2017human for detailed information concerning the CEFR-ASAG corpus.

@inproceedings{tack2017human,
  title={Human and automated CEFR-based grading of short answers},
  author={Tack, Ana{\"\i}s and Fran{\c{c}}ois, Thomas and Roekhaut, Sophie and Fairon, C{\'e}drick},
  booktitle={Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications},
  pages={169--179},
  year={2017}
}
