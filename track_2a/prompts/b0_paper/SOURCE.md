# Baseline prompt (b0)

`template_{de,fr,it}_admin.txt` are unchanged copies of `prompt_templates/template_{de,fr,it}_admin.txt`
from <https://github.com/ZurichNLP/SwissGov-RSD> (commit `1807a42100e742ed03d337c54c4b9ea86995f565`).
They are the few-shot prompt of the SwissGov-RSD paper (Wastl, Vamvas, Sennrich, ACL 2026, Appendix E):
the model labels every token of both documents with a similarity score from 0 to 5.

They are used here only to reproduce the paper's baseline prompting setup with Apertus, which is the
reference point named in the challenge description.

To settle before the repository is made public: the source repository has no licence file, so confirm
with the authors that these three files may be redistributed, or fetch them at run time instead.
